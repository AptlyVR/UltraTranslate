"""Who is speaking: local diarization inside each utterance + online speaker clustering + saved voiceprints.

* pyannote segmentation-3.0 (powerset, up to 3 local speakers incl. overlap) finds speaker changes and
  overlapped speech inside an utterance, so two people talking back-to-back become two lines.
* ECAPA-TDNN (speechbrain, VoxCeleb) embeddings identify each turn against this session's speakers
  ("Speaker 1", "Speaker 2", ...), creating a new one when nobody matches.
* Saved voiceprints (opt-in per speaker) auto-name people next time. Stored locally only.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from ..settings import APP_DIR, SPEAKERS_PATH, atomic_write_json
from .events import PALETTE, Speaker

SR = 16000
REAL_SCENE = os.environ.get("REAL_SCENE", "real_conversation")   # a real recording + its .diar.json


@dataclass
class Turn:
    start: int            # sample offsets inside the utterance
    end: int
    overlap: bool = False
    local: int = 0


@dataclass
class Voiceprint:
    id: str
    name: str
    embedding: np.ndarray
    count: int = 1
    seconds: float = 0.0
    language: str = ""
    updated: float = field(default_factory=time.time)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class VoiceprintStore:
    def __init__(self, path=SPEAKERS_PATH):
        self.path = path
        self.items: dict[str, Voiceprint] = {}
        self.lock = threading.Lock()
        self.load()

    def load(self) -> None:
        import json
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for r in raw.get("voiceprints", []):
            try:
                self.items[r["id"]] = Voiceprint(r["id"], r["name"], _unit(np.asarray(r["embedding"], np.float32)),
                                                 int(r.get("count", 1)), float(r.get("seconds", 0)),
                                                 r.get("language", ""), float(r.get("updated", 0)))
            except (KeyError, ValueError, TypeError):
                continue

    def save(self) -> None:
        with self.lock:
            atomic_write_json(self.path, {"model": "speechbrain/spkrec-ecapa-voxceleb", "voiceprints": [
                {"id": v.id, "name": v.name, "embedding": [round(float(x), 6) for x in v.embedding],
                 "count": v.count, "seconds": round(v.seconds, 1), "language": v.language, "updated": v.updated}
                for v in self.items.values()]})

    def match(self, emb: np.ndarray) -> tuple[Voiceprint | None, float, float]:
        if not self.items:
            return None, 0.0, 0.0
        scored = sorted(((float(v.embedding @ emb), v) for v in self.items.values()), key=lambda x: -x[0])
        best, vp = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        return vp, best, best - second

    def upsert(self, name: str, emb: np.ndarray, seconds: float, language: str, vid: str | None = None) -> Voiceprint:
        with self.lock:
            vp = self.items.get(vid or "") or next((v for v in self.items.values() if v.name == name), None)
            if vp is None:
                vp = Voiceprint(uuid.uuid4().hex[:12], name, _unit(emb), 1, seconds, language)
                self.items[vp.id] = vp
            else:
                w = min(vp.seconds, 120.0)
                vp.embedding = _unit(vp.embedding * w + emb * max(seconds, 1.0))
                vp.count += 1
                vp.seconds += seconds
                vp.name = name or vp.name
                vp.language = language or vp.language
                vp.updated = time.time()
        self.save()
        return vp

    def delete(self, vid: str) -> None:
        with self.lock:
            self.items.pop(vid, None)
        self.save()


class SpeakerEngine:
    def __init__(self, cfg, log=print, device: str = "cuda"):
        self.cfg = cfg
        self.log = log
        self.device = device
        self.seg_model = None
        self.powerset = None
        self.encoder = None
        self.redim = None                  # ReDimNet-b5 (second opinion for clustering), optional
        self.lock = threading.Lock()
        self.store = VoiceprintStore()
        self.win_cache: dict[str, dict[int, np.ndarray]] = {}
        self.trace = None                  # list -> assign() records its inputs (benches only)
        self.reset_session()

    # -- models -----------------------------------------------------------------------------------
    def load(self) -> None:
        import torch
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if self.device == "cpu":
            torch.set_num_threads(4)
        t0 = time.perf_counter()
        if self.seg_model is None and not getattr(self, "_seg_missing", False):
            try:
                from pyannote.audio import Model
                from pyannote.audio.utils.powerset import Powerset
                m = Model.from_pretrained("pyannote/segmentation-3.0")
                m.eval().to(torch.device(self.device))
                self.seg_model = m
                spec = m.specifications
                self.powerset = Powerset(len(spec.classes), spec.powerset_max_classes).to(torch.device(self.device))
            except Exception as exc:
                # gated model not downloaded (needs a free Hugging Face login): keep working on voice-embedding
                # change points alone, which separates back-to-back speakers somewhat less well
                self._seg_missing = True
                self.log(f"pyannote/segmentation-3.0 unavailable ({type(exc).__name__}); speaker changes from voice "
                         "embeddings only. Run install.cmd again to add it for better speaker separation.")
        if self.encoder is None:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(APP_DIR / "models" / "ecapa"),
                run_opts={"device": "cuda:0" if self.device == "cuda" else self.device}, local_strategy=LocalStrategy.COPY)
        if self.redim is None and getattr(self.cfg, "fuse_redimnet", False):
            try:
                hub = Path(torch.hub.get_dir()) / "IDRnD_ReDimNet_master"
                kw = dict(model_name="b5", train_type="ptn", dataset="vox2")
                m = (torch.hub.load(str(hub), "ReDimNet", source="local", **kw) if hub.is_dir() else
                     torch.hub.load("IDRnD/ReDimNet", "ReDimNet", trust_repo=True, **kw))
                self.redim = m.eval().to(torch.device(self.device))
            except Exception as exc:
                self.log(f"ReDimNet unavailable ({exc}); using ECAPA alone")
                self.redim = None
        # warm
        self.embed(np.zeros(SR, np.float32) + 1e-4)
        if self.redim is not None:
            self._redim_embed(np.zeros(SR, np.float32) + 1e-4)
        self.local_turns(np.zeros(SR * 2, np.float32) + 1e-4)
        self.log(f"Speaker models ready in {(time.perf_counter() - t0) * 1000:.0f} ms")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        import torch
        with self.lock, torch.inference_mode():
            wav = torch.from_numpy(np.ascontiguousarray(audio, np.float32)).unsqueeze(0).to(self.device)
            e = self.encoder.encode_batch(wav).squeeze().float().cpu().numpy()
        return _unit(e.astype(np.float32))

    def _redim_embed(self, audio: np.ndarray) -> np.ndarray:
        import torch
        x = audio[-SR * 10:] if len(audio) > SR * 10 else audio
        if len(x) < SR // 2:
            x = np.pad(x, (0, SR // 2 - len(x)))
        with self.lock, torch.inference_mode():
            e = self.redim(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(self.device))
        return _unit(e.squeeze().float().cpu().numpy().astype(np.float32))

    def _match_emb(self, audio: np.ndarray, emb: np.ndarray) -> np.ndarray:
        """Embedding used for clustering decisions: ECAPA alone, or ECAPA + ReDimNet fused."""
        if self.redim is None:
            return emb
        return _unit(np.concatenate([_unit(emb), self._redim_embed(audio)]).astype(np.float32))

    @property
    def _fused(self) -> bool:
        return self.redim is not None

    def _thr(self, name: str) -> float:
        if self._fused:
            return getattr(self.cfg, f"fused_{name}", getattr(self.cfg, name))
        return getattr(self.cfg, name)

    def embed_batch(self, windows: list[np.ndarray]) -> np.ndarray:
        import torch
        with self.lock, torch.inference_mode():
            wav = torch.from_numpy(np.stack(windows).astype(np.float32)).to(self.device)
            e = self.encoder.encode_batch(wav).squeeze(1).float().cpu().numpy()
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)

    # -- incremental window embeddings (computed while the person is still talking) ------------------
    WIN_S = 1.5

    def _hop(self) -> int:
        return int((0.375 if self.device != "cpu" else 0.75) * SR)

    def window_embs(self, audio: np.ndarray, uid: str = "") -> tuple[list[int], np.ndarray]:
        """Embeddings of all 1.5 s windows of `audio`; windows already embedded for this utterance (during
        live partials) are reused, so only the newest ones cost anything at commit time."""
        win, hop = int(self.WIN_S * SR), self._hop()
        starts = list(range(0, len(audio) - win + 1, hop))
        cache = self.win_cache.setdefault(uid, {}) if uid else {}
        missing = [st for st in starts if st not in cache]
        if missing:
            embs = self.embed_batch([audio[st:st + win] for st in missing])
            for st, e in zip(missing, embs):
                cache[st] = e
        while len(self.win_cache) > 6:
            self.win_cache.pop(next(iter(self.win_cache)))
        return starts, (np.stack([cache[st] for st in starts]) if starts else np.zeros((0, 192), np.float32))

    def precompute(self, uid: str, audio: np.ndarray) -> None:
        if self.encoder is not None and len(audio) >= int(self.WIN_S * SR):
            self.window_embs(audio, uid)

    def turn_embedding(self, uid: str, start: int, end: int) -> np.ndarray | None:
        """Mean of the cached window embeddings inside [start, end) - avoids re-embedding the whole turn."""
        cache = self.win_cache.get(uid) or {}
        win = int(self.WIN_S * SR)
        inside = [e for st, e in cache.items() if st >= start and st + win <= end]
        if len(inside) < 2:
            return None
        return _unit(np.mean(inside, axis=0).astype(np.float32))

    def change_points(self, audio: np.ndarray, uid: str = "") -> list[int]:
        """Speaker-change sample offsets from voice embeddings of adjacent 1.5 s windows.
        Catches back-to-back turns the segmentation model merges (different people, <0.3 s gap)."""
        n = len(audio)
        win, hop = int(self.WIN_S * SR), self._hop()
        if n < 2 * win + hop:
            return []
        starts, embs = self.window_embs(audio, uid)
        k = win // hop                       # compare non-overlapping neighbours
        score = np.array([float(embs[i] @ embs[i + k]) for i in range(len(starts) - k)])
        if score.size == 0:
            return []
        cuts = []
        thr = self.cfg.change_dip
        order = np.argsort(score)
        for i in order:
            if score[i] >= thr:
                break
            pos = starts[i] + win              # boundary between the two windows
            if all(abs(pos - c) >= int(1.5 * SR) for c in cuts) and win <= pos <= n - win:
                left, right = audio[max(0, pos - 3 * SR):pos], audio[pos:pos + 3 * SR]
                if min(len(left), len(right)) < win:
                    continue
                el, er = self.embed(left[-win * 2:]), self.embed(right[:win * 2])
                if float(el @ er) < self.cfg.change_confirm:
                    cuts.append(pos)
        # snap each cut to the quietest 20 ms within +-0.4 s
        snapped = []
        fr = int(0.02 * SR)
        for c in sorted(cuts):
            lo, hi = max(fr, c - int(0.4 * SR)), min(n - fr, c + int(0.4 * SR))
            if hi - lo < fr * 2:
                snapped.append(c)
                continue
            seg = audio[lo:hi]
            m = len(seg) // fr
            rms = np.sqrt((seg[:m * fr].reshape(m, fr) ** 2).mean(axis=1))
            snapped.append(lo + int(np.argmin(rms)) * fr + fr // 2)
        return snapped

    def split_turns(self, audio: np.ndarray, min_turn_s: float = 1.4, uid: str = "") -> list[Turn]:
        """Segmentation-model turns (overlap aware) + embedding change points, short pieces merged."""
        turns = self.local_turns(audio)
        cuts = sorted({t.start for t in turns[1:]} | set(self.change_points(audio, uid)))
        n = len(audio)
        bounds = [0] + [c for c in cuts if 0 < c < n] + [n]
        pieces = []
        for a, b in zip(bounds, bounds[1:]):
            ov = any(t.overlap and min(b, t.end) - max(a, t.start) > 0.5 * (b - a) for t in turns)
            pieces.append(Turn(a, b, ov))
        # merge pieces shorter than min_turn_s into the neighbour they sound most like (or the longer one)
        min_len = int(min_turn_s * SR)
        while len(pieces) > 1:
            i = min(range(len(pieces)), key=lambda j: pieces[j].end - pieces[j].start)
            if pieces[i].end - pieces[i].start >= min_len:
                break
            if i == 0:
                j = 1
            elif i == len(pieces) - 1:
                j = i - 1
            else:
                j = i - 1 if (pieces[i - 1].end - pieces[i - 1].start) >= (pieces[i + 1].end - pieces[i + 1].start) else i + 1
            lo, hi = min(i, j), max(i, j)
            pieces[lo] = Turn(pieces[lo].start, pieces[hi].end, pieces[lo].overlap or pieces[hi].overlap)
            del pieces[hi]
        return pieces

    def local_turns(self, audio: np.ndarray) -> list[Turn]:
        """Split an utterance into single-speaker turns using the powerset segmentation model."""
        import torch
        n = len(audio)
        if self.seg_model is None:              # model not installed: one turn; embedding change points still split it
            return [Turn(0, n)]
        win = SR * 10
        acts = []
        frames_per_sample = None
        with self.lock, torch.inference_mode():
            for start in range(0, n, win):
                chunk = audio[start:start + win]
                if len(chunk) < SR // 2 and acts:
                    # too short to analyse alone: extend the previous decision over it
                    extra = max(1, int(round(len(chunk) * frames_per_sample)))
                    acts.append(np.repeat(acts[-1][-1:], extra, axis=0))
                    break
                x = torch.from_numpy(np.ascontiguousarray(chunk, np.float32))[None, None].to(self.device)
                logits = self.seg_model(x)
                ml = self.powerset.to_multilabel(logits)[0].float().cpu().numpy()   # frames x 3
                acts.append(ml)
                if frames_per_sample is None:
                    frames_per_sample = ml.shape[0] / len(chunk)
        act = np.concatenate(acts, axis=0)
        if act.size == 0:
            return [Turn(0, n)]
        frames = act.shape[0]
        hop = n / frames
        # smooth: ~150 ms majority filter per class
        k = max(1, int(0.15 * SR / hop))
        if k > 1:
            kern = np.ones(k) / k
            act = np.stack([np.convolve(act[:, c], kern, mode="same") for c in range(act.shape[1])], 1) > 0.5
        else:
            act = act > 0.5
        active_total = act.sum(axis=0) * hop / SR
        speakers = [c for c in range(act.shape[1]) if active_total[c] >= 0.5]
        if len(speakers) <= 1:
            ov = False
            return [Turn(0, n, ov, speakers[0] if speakers else 0)]
        # dominant speaker per frame among active ones (ties -> previous)
        owner = np.full(frames, -1)
        prev = -1
        for f in range(frames):
            on = [c for c in speakers if act[f, c]]
            if not on:
                owner[f] = prev
            elif prev in on:
                owner[f] = prev
            else:
                owner[f] = on[0]
            prev = owner[f]
        overlap = act[:, speakers].sum(axis=1) >= 2
        # runs
        runs = []
        s = 0
        for f in range(1, frames + 1):
            if f == frames or owner[f] != owner[s]:
                runs.append([s, f, int(owner[s])])
                s = f
        # absorb short runs (< 0.45 s) into the longer neighbour
        min_frames = int(0.45 * SR / hop)
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for i, (a, b, o) in enumerate(runs):
                if b - a < min_frames or o < 0:
                    j = i - 1 if i > 0 and (i == len(runs) - 1 or (runs[i - 1][1] - runs[i - 1][0])
                                            >= (runs[i + 1][1] - runs[i + 1][0])) else i + 1
                    lo, hi = min(i, j), max(i, j)
                    runs[lo] = [runs[lo][0], runs[hi][1], runs[j][2]]
                    del runs[hi]
                    changed = True
                    break
            merged = [runs[0]]
            for r in runs[1:]:
                if r[2] == merged[-1][2]:
                    merged[-1][1] = r[1]
                else:
                    merged.append(r)
            runs = merged
        pad = int(0.08 * SR)
        turns = []
        for a, b, o in runs:
            sa, sb = int(a * hop), int(b * hop)
            ov = bool(overlap[a:b].mean() > 0.25) if b > a else False
            turns.append(Turn(max(0, sa - pad), min(n, sb + pad), ov, o))
        return turns

    # -- session clustering ------------------------------------------------------------------------
    def reset_session(self) -> None:
        self.speakers: dict[str, Speaker] = {}
        self.centroids: dict[str, np.ndarray] = {}
        self.weights: dict[str, float] = {}
        self.references: dict[str, tuple[float, np.ndarray]] = {}   # best clean clip per speaker
        self.ref_version: dict[str, int] = {}
        self.last_speaker = ""
        self._next = 1
        self.alias: dict[str, str] = {}       # merged-away speaker id -> the id it became
        self.clock = 0.0                      # seconds of speech assigned this session (the merge rules' clock)
        self.born: dict[str, float] = {}
        self.fcent: dict[str, np.ndarray] = {}  # clustering centroids (fused space); ECAPA ones stay for voiceprints
        self.on_merge = getattr(self, "on_merge", None)

    def resolve(self, sid: str) -> str:
        seen = 0
        while sid in self.alias and seen < 20:
            sid, seen = self.alias[sid], seen + 1
        return sid

    def _new_speaker(self) -> Speaker:
        sid = f"S{self._next}"
        sp = Speaker(sid, color=PALETTE[(self._next - 1) % len(PALETTE)])
        self._next += 1
        self.speakers[sid] = sp
        return sp

    def peek(self, emb: np.ndarray):
        """Best-matching session speaker without changing anything (for speculative work)."""
        best_id, best = "", -1.0
        for sid, c in self.centroids.items():
            sc = float(c @ emb)
            if sc > best:
                best_id, best = sid, sc
        return self.speakers.get(best_id) if best_id and best >= self.cfg.session_threshold else None

    def assign(self, audio: np.ndarray, overlap: bool = False, emb: np.ndarray | None = None
               ) -> tuple[Speaker, float]:
        """Return (speaker, score) for one single-speaker turn."""
        dur = len(audio) / SR
        self.clock += dur
        if dur < 0.5 and self.last_speaker in self.speakers:
            return self.speakers[self.last_speaker], 0.0
        if emb is None:
            emb = self.embed(audio)
        if self.trace is not None:         # bench: record every decision input for offline replay
            self.trace.append({"dur": dur, "overlap": bool(overlap), "emb": emb.tolist()})
        memb = self._match_emb(audio, emb)
        space = self.fcent if self._fused else self.centroids
        best_id, best = "", -1.0
        for sid, c in space.items():
            sc = float(c @ memb)
            if sc > best:
                best_id, best = sid, sc
        thr = self._thr("session_threshold") - (0.05 if dur < 1.5 else 0.0)
        if best_id and best >= thr:
            sp = self.speakers[best_id]
            if not overlap and dur >= 1.0:
                w = min(self.weights[best_id], 60.0)
                self.centroids[best_id] = _unit(self.centroids[best_id] * w + emb * dur)
                if self._fused:
                    self.fcent[best_id] = _unit(self.fcent[best_id] * w + memb * dur)
                self.weights[best_id] += dur
        elif len(self.speakers) >= self.cfg.max_speakers and best_id:
            sp = self.speakers[best_id]
        else:
            new_min = getattr(self.cfg, "new_min_s", 2.0)
            if dur < new_min and self.speakers:
                # too short (laugh, "un", reverb tail) to found a new voice reliably: nearest existing one,
                # or whoever spoke last when nothing is close; the centroid is not touched
                if not best_id or (best < thr - 0.15 and self.last_speaker in self.speakers):
                    best_id = self.last_speaker if self.last_speaker in self.speakers else best_id
                sp = self.speakers[best_id]
                sp.utterances += 1
                sp.seconds += dur
                sp.last_seen = time.time()
                self.last_speaker = sp.id
                return sp, best
            sp = self._new_speaker()
            self.born[sp.id] = self.clock
            self.centroids[sp.id] = emb
            if self._fused:
                self.fcent[sp.id] = memb
            self.weights[sp.id] = dur
            best = 1.0
        sp.utterances += 1
        sp.seconds += dur
        sp.last_seen = time.time()
        self.last_speaker = sp.id
        self._identify(sp)
        merged_into = self._merge_pass()
        if sp.id in merged_into:
            sp = self.speakers[self.resolve(sp.id)]
            self.last_speaker = sp.id
        return sp, best

    def _minor_candidate(self, space):
        """A label that has been around for minor_age_s of conversation but has under minor_total_s of talk is
        an over-split (laugh, wake word, reverb tail), not a person: fold it into the closest voice."""
        tot, age = getattr(self.cfg, "minor_total_s", 0.0), getattr(self.cfg, "minor_age_s", 30.0)
        if not tot or len(space) < 2:
            return None
        for sid in list(space):
            sp = self.speakers.get(sid)
            if sp is None or sp.name or sp.identity or sp.seconds >= tot:
                continue
            if self.clock - self.born.get(sid, 0.0) < age:
                continue
            others = [(float(space[sid] @ space[o]), o) for o in space if o != sid]
            sc, o = max(others)
            return (sc, o, sid) if self.weights.get(o, 0) >= self.weights.get(sid, 0) else (sc, sid, o)
        return None

    def _merge_pass(self) -> dict:
        """Two labels whose voice centroids have grown alike are one person split early on (a short first
        turn, reverb, a laugh). Merge the smaller into the larger; named/identified speakers keep their name.
        Returns {dropped id: kept id} for this pass."""
        thr = self._thr("merge_threshold")
        space = self.fcent if self._fused else self.centroids
        done = {}
        while True:
            ids = [s for s in space if self.weights.get(s, 0) > 0]
            best = None
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    na, nb = self.speakers[a].name, self.speakers[b].name
                    if na and nb and na != nb:
                        continue                         # the user said these are different people
                    sc = float(space[a] @ space[b])
                    if sc >= thr and (best is None or sc > best[0]):
                        best = (sc, a, b)
            if best is None:
                best = self._minor_candidate(space)
            if best is None:
                return done
            _, a, b = best
            sa, sb = self.speakers[a], self.speakers[b]
            keep, drop = (a, b) if (bool(sa.name) or bool(sa.identity), self.weights[a]) >=                 (bool(sb.name) or bool(sb.identity), self.weights[b]) else (b, a)
            if not self.speakers[keep].name and self.speakers[drop].name:
                self.speakers[keep].name = self.speakers[drop].name
            self.merge(keep, drop)
            self.alias[drop] = keep
            done[drop] = keep
            self.log(f"Speakers: {drop} is the same voice as {keep} (similarity {best[0]:.2f}) - merged")
            if self.on_merge:
                try:
                    self.on_merge(drop, keep)
                except Exception:
                    pass

    def offer_reference(self, sid: str, audio: np.ndarray, avg_logprob: float, overlap: bool) -> None:
        """Keep each speaker's clearest clip for cloning their voice: 1.2-12 s, no overlap, confidently
        recognised. Score = seconds x confidence; a clearly better clip replaces the old one (version bumps)."""
        dur = len(audio) / SR
        if overlap or not 1.2 <= dur <= 12:
            return
        conf = float(np.clip(1.0 + avg_logprob, 0.05, 1.0))
        score = min(dur, 8.0) * conf
        old = self.references.get(sid)
        if old is None or score > old[0] * 1.15:
            self.ref_version[sid] = self.ref_version.get(sid, 0) + 1
            self.references[sid] = (score, audio.copy())

    def _identify(self, sp: Speaker) -> None:
        if sp.name and not sp.identity:
            return  # user-named and not from a voiceprint: leave it alone
        if self.weights.get(sp.id, 0) < 2.0:
            return
        vp, score, margin = self.store.match(self.centroids[sp.id])
        if vp and score >= self.cfg.identity_threshold and margin >= 0.05:
            taken = any(o.identity == vp.id and o.id != sp.id for o in self.speakers.values())
            if not taken:
                sp.identity, sp.name, sp.match_score = vp.id, vp.name, score
                if vp.language and not sp.language:
                    sp.language = vp.language
                if self.cfg.learn_voiceprints and score >= self.cfg.identity_threshold + 0.1:
                    self.store.upsert(vp.name, self.centroids[sp.id], self.weights[sp.id], sp.language, vp.id)

    # -- user actions ------------------------------------------------------------------------------
    def rename(self, sid: str, name: str) -> None:
        sp = self.speakers.get(sid)
        if sp:
            sp.name = name.strip()
            if sp.identity and sp.identity in self.store.items:
                self.store.upsert(sp.name, self.centroids[sid], 0.0, sp.language, sp.identity)

    def save_voiceprint(self, sid: str) -> bool:
        sp = self.speakers.get(sid)
        if not sp or sid not in self.centroids or not sp.name:
            return False
        vp = self.store.upsert(sp.name, self.centroids[sid], self.weights[sid], sp.language, sp.identity or None)
        sp.identity = vp.id
        return True

    def forget_voiceprint(self, sid: str) -> None:
        sp = self.speakers.get(sid)
        if sp and sp.identity:
            self.store.delete(sp.identity)
            sp.identity = ""

    def merge(self, keep: str, drop: str) -> None:
        if keep == drop or keep not in self.speakers or drop not in self.speakers:
            return
        wk, wd = self.weights[keep], self.weights[drop]
        self.centroids[keep] = _unit(self.centroids[keep] * wk + self.centroids[drop] * wd)
        if keep in self.fcent and drop in self.fcent:
            self.fcent[keep] = _unit(self.fcent[keep] * wk + self.fcent[drop] * wd)
        self.fcent.pop(drop, None)
        self.weights[keep] = wk + wd
        a, b = self.speakers[keep], self.speakers.pop(drop)
        a.utterances += b.utterances
        a.seconds += b.seconds
        for k, v in b.lang_votes.items():
            a.lang_votes[k] = a.lang_votes.get(k, 0) + v
        self.centroids.pop(drop, None)
        self.weights.pop(drop, None)
        ref = self.references.pop(drop, None)
        if ref and ref[0] > self.references.get(keep, (0, None))[0]:
            self.references[keep] = ref
        if self.last_speaker == drop:
            self.last_speaker = keep
