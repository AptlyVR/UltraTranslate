"""CUDA-graph streaming decode for VoxCPM2 (same weights, same sampling).

Why: eager VoxCPM2 spends most of each 160 ms audio patch on overhead, not
math. Every patch launches thousands of small kernels (seven CFG passes of the
12-layer LocDiT, the 12-layer patch encoder, the 28-layer LM and the 8-layer
residual LM), and every LM layer attends over the whole 8,192-slot static KV
cache through PyTorch's MATH attention kernel because grouped-query attention
plus a mask disables the fused kernels (~0.9 ms per layer on an RTX 3080).

What this does:
* Patch work is replayed as CUDA graphs: DiT+encoder (``G_dit``), the
  LM/residual-LM update (``G_lm``) and the VAE decode (one graph per window).
* Audio is copied out right after ``G_dit``/VAE; ``G_lm`` for the next patch is
  queued before the CPU waits, so the GPU never idles between patches.
* LM decode attention uses FlashAttention's KV-cache kernel over only the valid
  positions (GQA native). Without flash_attn it falls back to the stock math.
* VAE weight-norm is folded once (bit-identical weights).

Sampling, steps, CFG, prefix length and prompt handling are unchanged. DiT,
encoder and VAE kernels are the eager ones, so with the math fallback the audio
is bit-identical to stock; with FlashAttention only the LM attention reduction
order differs (bf16 rounding level).
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Iterator, Optional

import torch
from einops import rearrange

try:  # Linux/.3: flash-attn 2.7 is installed in the server venv.
    from flash_attn import flash_attn_with_kvcache
except Exception:  # pragma: no cover - Windows test boxes
    flash_attn_with_kvcache = None

MAX_WINDOW = 8          # server clamps streaming_prefix_len to 2..8
KV_SLOTS = 4096         # prompt (<~600) + text + max 768 generated patches
SUFFIX_BUCKETS = (16, 32, 64, 128, 256)  # per-request tail lengths captured as graphs


def fold_weight_norm(module: torch.nn.Module) -> int:
    """Compute weight-norm weights once instead of on every forward (same values)."""
    folded = 0
    for sub in module.modules():
        for hook in list(getattr(sub, "_forward_pre_hooks", {}).values()):
            if hook.__class__.__name__ == "WeightNorm":
                torch.nn.utils.remove_weight_norm(sub, name=hook.name)
                folded += 1
                break
    return folded


def _rope(q, k, cos, sin):
    """Same math as voxcpm.modules.minicpm4.model.apply_rotary_pos_emb."""
    dtype = q.dtype
    q, k = q.float(), k.float()

    def rot(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos + rot(q) * sin).to(dtype), (k * cos + rot(k) * sin).to(dtype)


def _rms(x, w, eps):
    """Same math as voxcpm.modules.minicpm4.model.rms_layernorm."""
    dt = x.dtype
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)).to(dt) * w


def _pre(h, ln, eps, wq, wk, wv, cos, sin, nh: int, nkv: int, hd: int):
    """Norm + QKV + RoPE for B tokens. Returns BSHD q/k/v (flash layout)."""
    B = h.shape[1]
    hn = _rms(h, ln, eps)
    q = torch.nn.functional.linear(hn, wq).view(1, B, nh, hd).transpose(1, 2)
    k = torch.nn.functional.linear(hn, wk).view(1, B, nkv, hd).transpose(1, 2)
    v = torch.nn.functional.linear(hn, wv).view(1, B, nkv, hd)
    if cos is not None:
        q, k = _rope(q, k, cos, sin)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.contiguous()


def _post(h, o, wo, ln, eps, wg, wu, wd, scale: float):
    """o_proj + residual + norm + SwiGLU MLP + residual (MiniCPM decoder layer tail)."""
    a = torch.nn.functional.linear(o, wo)
    h = h + (a * scale if scale != 1.0 else a)
    hn = _rms(h, ln, eps)
    m = torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(hn, wg))
                                   * torch.nn.functional.linear(hn, wu), wd)
    return h + (m * scale if scale != 1.0 else m)


# Distinct code objects so each compile mode keeps its own Dynamo cache.
def _pre_dec(*a): return _pre(*a)
def _post_dec(*a): return _post(*a)
def _pre_pf(*a): return _pre(*a)
def _post_pf(*a): return _post(*a)


class _LMDecoder:
    """Decode/prefill for a MiniCPMModel against a BSHD KV cache (FlashAttention).

    Layer math is expressed as two functional blocks over plain tensors, so one
    compiled kernel set serves every layer of both LMs (no per-module guards).
    """

    def __init__(self, lm, device, dtype, use_flash: bool, kernels=None):
        cfg = lm.config
        self.lm = lm
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.kv_channels or cfg.hidden_size // cfg.num_attention_heads
        n = cfg.num_hidden_layers
        shape = (n, 1, KV_SLOTS, self.nkv, self.hd)
        numel = n * KV_SLOTS * self.nkv * self.hd
        stock = getattr(getattr(lm, "kv_cache", None), "kv_cache", None)
        if stock is not None and stock.dtype == dtype and stock.is_contiguous() and stock.numel() >= 2 * numel:
            # Share the stock model's idle static cache (8,192 slots, BHSD) instead of
            # allocating more VRAM next to ASR. Only one path runs at a time on the
            # model thread and each request reloads its own cache contents.
            flat = stock.view(-1)
            self.k, self.v = flat[:numel].view(shape), flat[numel:2 * numel].view(shape)
        else:
            self.k = torch.zeros(shape, device=device, dtype=dtype)
            self.v = torch.zeros_like(self.k)
        self.seqlens = torch.zeros((1,), device=device, dtype=torch.int32)
        self.pos = torch.zeros((1,), device=device, dtype=torch.long)
        self.use_flash = use_flash
        self.slots = torch.arange(KV_SLOTS, device=device)
        self.kernels = kernels or {}
        self.layers = []
        for layer in lm.layers:
            at, ml = layer.self_attn, layer.mlp
            scale = (layer.scale_depth / layer.num_hidden_layers ** 0.5) if layer.use_mup else 1.0
            self.layers.append(dict(
                pre=(layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon,
                     at.q_proj.weight, at.k_proj.weight, at.v_proj.weight),
                post=(at.o_proj.weight, layer.post_attention_layernorm.weight,
                      layer.post_attention_layernorm.variance_epsilon,
                      ml.gate_proj.weight, ml.up_proj.weight, ml.down_proj.weight, float(scale))))

    def load(self, kv_tuples, length: int) -> None:
        for i, (k, v) in enumerate(kv_tuples):  # stock prefill: [1, nkv, T, hd]
            self.k[i, :, :length] = k.transpose(1, 2)
            self.v[i, :, :length] = v.transpose(1, 2)
        self.seqlens.fill_(length)
        self.pos.fill_(length)

    def load_prefix(self, k, v, length: int) -> None:
        """k, v: [layers, 1, length, nkv, hd] cached for one reference voice."""
        if length:
            self.k[:, :, :length].copy_(k)
            self.v[:, :, :length].copy_(v)
        self.seqlens.fill_(length)
        self.pos.fill_(length)

    def _attend(self, i, q, k, v, positions):
        B = q.shape[1]
        if self.use_flash:
            return flash_attn_with_kvcache(q, self.k[i], self.v[i], k=k, v=v,
                                           cache_seqlens=self.seqlens, causal=True)
        kc, vc = self.k[i], self.v[i]
        kc.index_copy_(1, positions, k)
        vc.index_copy_(1, positions, v)
        mask = self.slots[None, :] <= positions[:, None]
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), kc.transpose(1, 2), vc.transpose(1, 2), attn_mask=mask, enable_gqa=True)
        return out.transpose(1, 2)

    def run(self, x, n_real=None, *, prefill=False):
        """x: [1, B, H]. Writes KV at [pos, pos+B); advances by n_real (default B)."""
        lm, B = self.lm, x.shape[1]
        positions = self.pos + torch.arange(B, device=x.device)
        cos = sin = None
        if lm.rope_emb is not None:
            cos, sin = lm.rope_emb(positions)
        pre = self.kernels.get("pre_pf" if prefill else "pre_dec", _pre)
        post = self.kernels.get("post_pf" if prefill else "post_dec", _post)
        h = x
        for i, w in enumerate(self.layers):
            q, k, v = pre(h, *w["pre"], cos, sin, self.nh, self.nkv, self.hd)
            o = self._attend(i, q, k, v, positions).reshape(1, B, self.nh * self.hd)
            h = post(h, o, *w["post"])
        if n_real is None:
            self.seqlens.add_(B)
            self.pos.add_(B)
        else:
            self.seqlens.add_(n_real.to(torch.int32))
            self.pos.add_(n_real)
        return _rms(h, lm.norm.weight, lm.norm.variance_epsilon)

    def step(self, x):
        return self.run(x.unsqueeze(1))[:, 0]

    def tokens(self, x, n_real):
        return self.run(x, n_real, prefill=True)


class _ModuleShim(torch.nn.Module):
    """nn.Module wrapper around a compiled callable; other attributes read through to the original module."""

    def __init__(self, fn, orig):
        super().__init__()
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_orig", orig)

    def forward(self, *args, **kwargs):
        return object.__getattribute__(self, "_fn")(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_orig"), name)


class VoxCPM2FastPath:
    """Graph-captured streaming generation for one VoxCPM2Model.

    Capture and replay must happen on the model-owning thread.
    """

    def __init__(self, tts_model, *, fold_vae: bool = True, use_flash: Optional[bool] = None,
                 compile_mode: Optional[str] = None):
        self.m = tts_model
        self.device = torch.device(tts_model.device)
        self.dtype = tts_model._dtype()
        self.P = int(tts_model.patch_size)
        self.D = int(tts_model.audio_vae.latent_dim)
        self.H = int(tts_model.config.lm_config.hidden_size)
        self.L = int(tts_model.patch_size * tts_model._decode_chunk_size)
        self.folded = fold_weight_norm(tts_model.audio_vae) if fold_vae else 0
        self.use_flash = (flash_attn_with_kvcache is not None) if use_flash is None else bool(use_flash)
        self._lock = threading.Lock()
        self._pool = None
        self._dits: dict[tuple, torch.cuda.CUDAGraph] = {}
        self._vaes: dict[int, torch.cuda.CUDAGraph] = {}
        self._g_lm = None
        self._suffixes: dict[int, dict] = {}
        self._prefixes: "OrderedDict[int, tuple]" = OrderedDict()
        self.use_prefix = True
        dev, dt, H, P, D = self.device, self.dtype, self.H, self.P, self.D
        # Kernel fusion: Inductor-compiled DiT/encoder and LM layer blocks, replayed
        # inside the CUDA graphs. "off" keeps the eager kernels (bit-exact path).
        mode = compile_mode or os.environ.get("VOXCPM_FASTPATH_COMPILE", "max-autotune-no-cudagraphs")
        self.compile_mode = None if str(mode).lower() in ("", "0", "off", "none", "false") else mode
        kernels, self.est_c, self.enc_c = {}, None, None
        self._compiled_keys: set = set()  # only warmed (steps, cfg) pairs use the compiled DiT
        if self.compile_mode:
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = max(64, dynamo.config.cache_size_limit)
            c = lambda fn, m: torch.compile(fn, mode=m, dynamic=False)
            kernels = dict(pre_dec=c(_pre_dec, self.compile_mode), post_dec=c(_post_dec, self.compile_mode),
                           pre_pf=c(_pre_pf, "default"), post_pf=c(_post_pf, "default"))
            # newer torch.compile returns a plain function for a module; wrap it so it can stand in as
            # feat_decoder.estimator (a child-module slot) while the DiT graph is recorded
            self.est_c = _ModuleShim(c(tts_model.feat_decoder.estimator, self.compile_mode),
                                     tts_model.feat_decoder.estimator)
            self.enc_c = c(tts_model.feat_encoder, self.compile_mode)
        with torch.inference_mode():
            self.base = _LMDecoder(tts_model.base_lm, dev, dt, self.use_flash, kernels)
            self.res = _LMDecoder(tts_model.residual_lm, dev, dt, self.use_flash, kernels)
            # Static graph buffers. Graphs read raw addresses: all must outlive them.
            self.b_lm = torch.zeros((1, H), device=dev, dtype=dt)
            self.b_res = torch.zeros((1, H), device=dev, dtype=dt)
            self.b_cond = torch.zeros((1, P, D), device=dev, dtype=dt)
            self.b_curr = torch.zeros((1, H), device=dev, dtype=dt)
            self.b_stop = torch.zeros((1,), device=dev, dtype=torch.long)
            self.win = torch.zeros((1, D, P * MAX_WINDOW), device=dev, dtype=dt)
            self.audio = torch.zeros((self.L,), device=dev, dtype=torch.float32)
            vae = tts_model.audio_vae
            self.sr_cond = (torch.tensor([vae.out_sample_rate], device=dev, dtype=torch.int32)
                            if getattr(vae, "sr_bin_boundaries", None) is not None else None)
        self.host_audio = torch.empty((self.L,), dtype=torch.float32, pin_memory=True)
        self.host_stop = torch.empty((1,), dtype=torch.int64, pin_memory=True)
        self.ready = torch.cuda.Event()

    # ------------------------------------------------------------------ #
    def _capture(self, body, warmup: int = 3) -> torch.cuda.CUDAGraph:
        rng = torch.cuda.get_rng_state(self.device)
        try:
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    body()
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
                body()
            torch.cuda.synchronize(self.device)
        finally:
            torch.cuda.set_rng_state(rng, self.device)  # warmup consumed random numbers
        if self._pool is None:
            self._pool = graph.pool()
        return graph

    def _dit(self, n_timesteps: int, cfg_value: float) -> torch.cuda.CUDAGraph:
        key = (int(n_timesteps), float(cfg_value))
        if key in self._dits:
            return self._dits[key]
        m, P = self.m, self.P

        fd = m.feat_decoder

        def body():
            dit_hidden = torch.cat((m.lm_to_dit_proj(self.b_lm), m.res_to_dit_proj(self.b_res)), dim=-1)
            estimator = fd.estimator
            if self.est_c is not None and key in self._compiled_keys:  # swapped only while recording
                fd.estimator = self.est_c
            try:
                pred = fd(mu=dit_hidden, patch_size=P, cond=self.b_cond.transpose(1, 2).contiguous(),
                          n_timesteps=key[0], cfg_value=key[1]).transpose(1, 2)
            finally:
                fd.estimator = estimator
            encoder = self.enc_c if self.enc_c is not None else m.feat_encoder
            self.b_curr.copy_(m.enc_to_lm_proj(encoder(pred.unsqueeze(1)))[:, 0, :])
            self.b_stop.copy_(m.stop_head(m.stop_actn(m.stop_proj(self.b_lm))).argmax(dim=-1))
            self.b_cond.copy_(pred)
            self.win.copy_(torch.roll(self.win, -P, dims=-1))
            self.win[..., -P:] = pred.transpose(1, 2)

        self._dits[key] = self._capture(body)
        return self._dits[key]

    def _lm(self) -> torch.cuda.CUDAGraph:
        if self._g_lm is not None:
            return self._g_lm
        m = self.m

        def body():
            new_lm = m.fsq_layer(self.base.step(self.b_curr))
            new_res = self.res.step(m.fusion_concat_proj(torch.cat((new_lm, self.b_curr), dim=-1)))
            self.b_lm.copy_(new_lm)
            self.b_res.copy_(new_res)

        # Warmup writes throwaway rows near the end; every request reloads the cache.
        for dec in (self.base, self.res):
            dec.seqlens.fill_(KV_SLOTS - 8)
            dec.pos.fill_(KV_SLOTS - 8)
        self._g_lm = self._capture(body)
        return self._g_lm

    def _vae(self, n: int) -> torch.cuda.CUDAGraph:
        if n in self._vaes:
            return self._vaes[n]
        view, vae, L = self.win[..., -self.P * n:], self.m.audio_vae, self.L

        def body():
            self.audio.copy_(vae.decode(view.to(torch.float32), self.sr_cond)[..., -L:].reshape(-1))

        self._vaes[n] = self._capture(body)
        return self._vaes[n]

    def _suffix(self, bucket: int) -> dict:
        """Graph for prefilling `bucket` (padded) positions after a cached voice prefix."""
        if bucket in self._suffixes:
            return self._suffixes[bucket]
        m, dev, dt, H = self.m, self.device, self.dtype, self.H
        s = dict(
            x=torch.zeros((1, bucket, H), device=dev, dtype=dt),
            fe=torch.zeros((1, bucket, H), device=dev, dtype=dt),
            audio=torch.zeros((1, bucket, 1), device=dev, dtype=dt),
            text=torch.zeros((1, bucket, 1), device=dev, dtype=dt),
            n=torch.ones((1,), device=dev, dtype=torch.long),
            last=torch.zeros((1,), device=dev, dtype=torch.long),
        )

        def body():
            enc = self.base.tokens(s["x"], s["n"])
            enc = m.fsq_layer(enc) * s["audio"] + enc * s["text"]
            res = self.res.tokens(m.fusion_concat_proj(torch.cat((enc, s["audio"] * s["fe"]), dim=-1)), s["n"])
            self.b_lm.copy_(enc.index_select(1, s["last"])[:, 0])
            self.b_res.copy_(res.index_select(1, s["last"])[:, 0])

        for dec in (self.base, self.res):
            dec.seqlens.fill_(KV_SLOTS - bucket - 8)
            dec.pos.fill_(KV_SLOTS - bucket - 8)
        s["graph"] = self._capture(body)
        self._suffixes[bucket] = s
        return s

    def prepare(self, prompt_cache: dict) -> Optional[dict]:
        """Precompute the reference-voice prefix (KV + embeddings) once per voice."""
        mode = prompt_cache.get("mode") if isinstance(prompt_cache, dict) else None
        if mode not in ("reference", "ref_continuation") or "ref_audio_feat" not in prompt_cache:
            return None
        key = id(prompt_cache)
        hit = self._prefixes.get(key)
        if hit is not None and hit[0] is prompt_cache:
            self._prefixes.move_to_end(key)
            return hit[1]
        m, dev, dt = self.m, self.device, self.dtype
        with torch.inference_mode():
            rt, rf, rtm, ram = m._make_ref_prefix(prompt_cache["ref_audio_feat"], torch.device("cpu"))
            tok = rt.unsqueeze(0).to(dev)
            feat = rf.unsqueeze(0).to(dev).to(dt)
            tm = rtm.unsqueeze(0).to(dev)
            am = ram.unsqueeze(0).to(dev)
            fe = m.enc_to_lm_proj(m.feat_encoder(feat))
            scale = m.config.lm_config.scale_emb if m.config.lm_config.use_mup else 1.0
            comb = tm.unsqueeze(-1) * (m.base_lm.embed_tokens(tok) * scale) + am.unsqueeze(-1) * fe
            enc, kv = m.base_lm(inputs_embeds=comb, is_causal=True)
            enc = m.fsq_layer(enc) * am.unsqueeze(-1) + enc * tm.unsqueeze(-1)
            _, rkv = m.residual_lm(
                inputs_embeds=m.fusion_concat_proj(torch.cat((enc, am.unsqueeze(-1) * fe), dim=-1)),
                is_causal=True)
            state = dict(
                length=int(tok.shape[1]),
                bk=torch.stack([k.transpose(1, 2) for k, _ in kv]).contiguous(),
                bv=torch.stack([v.transpose(1, 2) for _, v in kv]).contiguous(),
                rk=torch.stack([k.transpose(1, 2) for k, _ in rkv]).contiguous(),
                rv=torch.stack([v.transpose(1, 2) for _, v in rkv]).contiguous(),
            )
            if mode == "ref_continuation":
                paf = prompt_cache["audio_feat"].unsqueeze(0).to(dev).to(dt)
                state["prompt_feat"] = paf
                state["prompt_embed"] = m.enc_to_lm_proj(m.feat_encoder(paf))
        self._prefixes[key] = (prompt_cache, state)
        while len(self._prefixes) > 8:
            self._prefixes.popitem(last=False)
        return state

    def warm(self, n_timesteps: int, cfg_value: float, windows=(1, 2, 3, 4),
             buckets=SUFFIX_BUCKETS) -> None:
        """Capture graphs (and compile kernels) up front. Pairs captured later on
        demand use eager DiT kernels so a request never waits for a compile."""
        with self._lock, torch.inference_mode():
            self._compiled_keys.add((int(n_timesteps), float(cfg_value)))
            self._dit(n_timesteps, cfg_value)
            self._lm()
            for n in windows:
                self._vae(int(n))
            for b in buckets:
                self._suffix(int(b))

    def _prefill_prefixed(self, state: dict, target_text: str, prompt_cache: dict):
        """Load the cached voice prefix, then prefill only the request-specific tail."""
        m, dev, dt = self.m, self.device, self.dtype
        prompt_text = prompt_cache.get("prompt_text", "") if prompt_cache.get("mode") == "ref_continuation" else ""
        tok = m.text_tokenizer(prompt_text + target_text) + [m.audio_start_token]
        n_text = len(tok)
        x_text = m.base_lm.embed_tokens(torch.tensor([tok], device=dev, dtype=torch.long))
        if m.config.lm_config.use_mup:
            x_text = x_text * m.config.lm_config.scale_emb
        if "prompt_embed" in state:
            n_audio = state["prompt_embed"].shape[1]
            x = torch.cat((x_text.to(dt), state["prompt_embed"]), dim=1)
            fe = torch.cat((torch.zeros_like(x_text, dtype=dt), state["prompt_embed"]), dim=1)
        else:
            n_audio = 0
            x, fe = x_text.to(dt), None
        n = n_text + n_audio
        if state["length"] + max(n, SUFFIX_BUCKETS[-1]) + 16 >= KV_SLOTS:
            raise ValueError("prompt too long for the fast path")
        self.base.load_prefix(state["bk"], state["bv"], state["length"])
        self.res.load_prefix(state["rk"], state["rv"], state["length"])
        bucket = next((b for b in SUFFIX_BUCKETS if b >= n), None)
        if bucket is not None and bucket in self._suffixes:
            s = self._suffixes[bucket]
            s["x"].zero_(); s["fe"].zero_(); s["audio"].zero_(); s["text"].zero_()
            s["x"][:, :n] = x
            if fe is not None:
                s["fe"][:, :n] = fe
            s["text"][:, :n_text] = 1
            s["audio"][:, n_text:n] = 1
            s["n"].fill_(n)
            s["last"].fill_(n - 1)
            s["graph"].replay()
        else:  # longer than the largest captured bucket: same math, eager
            nt = torch.tensor([n], device=dev, dtype=torch.long)
            a = torch.zeros((1, n, 1), device=dev, dtype=dt); a[:, n_text:] = 1
            t = 1 - a
            enc = self.base.tokens(x, nt)
            enc = m.fsq_layer(enc) * a + enc * t
            fe_all = fe if fe is not None else torch.zeros_like(x)
            res = self.res.tokens(m.fusion_concat_proj(torch.cat((enc, a * fe_all), dim=-1)), nt)
            self.b_lm.copy_(enc[:, -1]); self.b_res.copy_(res[:, -1])
        T = state["length"] + n
        if "prompt_feat" in state:
            self.b_cond.copy_(state["prompt_feat"][:, -1])
            return list(state["prompt_feat"].split(1, dim=1)), T
        self.b_cond.zero_()
        return [], T

    # ------------------------------------------------------------------ #
    def _inputs(self, target_text: str, prompt_cache: Optional[dict]):
        """Same sequence layout as VoxCPM2Model._generate_with_prompt_cache."""
        m, P, D = self.m, self.P, self.D
        mode = "zero_shot" if prompt_cache is None else prompt_cache.get("mode", "continuation")
        text = (prompt_cache.get("prompt_text", "") + target_text) if mode in ("continuation", "ref_continuation") else target_text
        tok = torch.LongTensor(m.text_tokenizer(text))
        tok = torch.cat([tok, torch.tensor([m.audio_start_token], dtype=torch.int32)], dim=-1)
        n_text = tok.shape[0]
        zf = lambda n: torch.zeros((n, P, D), dtype=torch.float32)
        ones = lambda n: torch.ones(n, dtype=torch.int32)
        zeros = lambda n: torch.zeros(n, dtype=torch.int32)
        if mode in ("zero_shot", "continuation"):
            paf = prompt_cache["audio_feat"] if prompt_cache else torch.empty((0, P, D), dtype=torch.float32)
            n = paf.size(0)
            tok = torch.cat([tok, torch.zeros(n, dtype=torch.int32)])
            feat = torch.cat([zf(n_text), paf], dim=0)
            tmask, amask = torch.cat([ones(n_text), zeros(n)]), torch.cat([zeros(n_text), ones(n)])
        elif mode == "reference":
            rt, rf, rtm, ram = m._make_ref_prefix(prompt_cache["ref_audio_feat"], tok.device)
            tok = torch.cat([rt, tok])
            feat = torch.cat([rf, zf(n_text)], dim=0)
            tmask, amask = torch.cat([rtm, ones(n_text)]), torch.cat([ram, zeros(n_text)])
        else:
            paf = prompt_cache["audio_feat"]
            n = paf.size(0)
            rt, rf, rtm, ram = m._make_ref_prefix(prompt_cache["ref_audio_feat"], tok.device)
            tok = torch.cat([rt, tok, torch.zeros(n, dtype=torch.int32)])
            feat = torch.cat([rf, zf(n_text), paf], dim=0)
            tmask = torch.cat([rtm, ones(n_text), zeros(n)])
            amask = torch.cat([ram, zeros(n_text), ones(n)])
        dev = self.device
        return (tok.unsqueeze(0).to(dev), tmask.unsqueeze(0).to(dev),
                feat.unsqueeze(0).to(dev).to(self.dtype), amask.unsqueeze(0).to(dev),
                len(m.text_tokenizer(target_text)))

    @torch.inference_mode()
    def generate(self, target_text: str, prompt_cache: Optional[dict], *, min_len: int = 2,
                 max_len: int = 2000, inference_timesteps: int = 10, cfg_value: float = 2.0,
                 streaming_prefix_len: int = 4, ratio_threshold: float = 6.0) -> Iterator[torch.Tensor]:
        """Yield 1-D float32 CPU tensors, one 160 ms patch each (stock streaming semantics)."""
        m, P = self.m, self.P
        with self._lock:
            window = max(1, min(int(streaming_prefix_len), MAX_WINDOW))
            g_dit, g_lm = self._dit(inference_timesteps, cfg_value), self._lm()
            vaes = [None] + [self._vae(n) for n in range(1, window + 1)]
            state = self.prepare(prompt_cache) if (self.use_prefix and prompt_cache is not None) else None
            if state is not None:
                # ---- prefill only the request tail after the cached voice prefix ----
                context, T = self._prefill_prefixed(state, target_text, prompt_cache)
                context = context[-(window - 1):] if window > 1 else []
                target_len = len(m.text_tokenizer(target_text))
            else:
                # ---- full prefill: same ops as VoxCPM2Model._inference ----
                text, tmask, feat, fmask, target_len = self._inputs(target_text, prompt_cache)
                T = text.shape[1]
                if T + 16 >= KV_SLOTS:
                    raise ValueError("prompt too long for the fast path")
                feat_embed = m.enc_to_lm_proj(m.feat_encoder(feat))
                scale = m.config.lm_config.scale_emb if m.config.lm_config.use_mup else 1.0
                combined = tmask.unsqueeze(-1) * (m.base_lm.embed_tokens(text) * scale) + fmask.unsqueeze(-1) * feat_embed
                context = []
                if fmask[0, -1].item() == 1:
                    idx = fmask.squeeze(0).nonzero(as_tuple=True)[0]
                    ctx = min(window - 1, len(idx))
                    if ctx > 0:
                        context = list(feat[:, idx[-ctx:], :, :].split(1, dim=1))
                enc, kv = m.base_lm(inputs_embeds=combined, is_causal=True)
                enc = m.fsq_layer(enc) * fmask.unsqueeze(-1) + enc * tmask.unsqueeze(-1)
                res, rkv = m.residual_lm(
                    inputs_embeds=m.fusion_concat_proj(torch.cat((enc, fmask.unsqueeze(-1) * feat_embed), dim=-1)),
                    is_causal=True)
                self.base.load(kv, T)
                self.res.load(rkv, T)
                self.b_lm.copy_(enc[:, -1, :])
                self.b_res.copy_(res[:, -1, :])
                self.b_cond.copy_(feat[:, -1, ...])
            max_len = min(int(target_len * ratio_threshold + 10), int(max_len), KV_SLOTS - T - 8)
            self.win.zero_()
            for c in context:
                self.win.copy_(torch.roll(self.win, -P, dims=-1))
                self.win[..., -P:] = rearrange(c, "b t p d -> b d (t p)")
            n_win = len(context)
            stream = torch.cuda.current_stream(self.device)
            for i in range(max_len):
                g_dit.replay()
                n_win = min(n_win + 1, window)
                vaes[n_win].replay()
                self.host_audio.copy_(self.audio, non_blocking=True)
                self.host_stop.copy_(self.b_stop, non_blocking=True)
                self.ready.record(stream)
                g_lm.replay()  # next patch's LM update runs while this chunk is delivered
                self.ready.synchronize()
                yield self.host_audio.clone()
                if i > min_len and int(self.host_stop[0]) == 1:
                    break
            stream.synchronize()
