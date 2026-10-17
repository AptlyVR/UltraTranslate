from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from concurrent.futures import CancelledError, TimeoutError as FutureTimeoutError
import gc
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

# Expandable segments failed CUDA-graph checkpoint restoration in this runtime.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
os.environ.setdefault("TQDM_DISABLE", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
# Compiled fast-path kernels (Inductor/Triton autotune) persist across restarts.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT_DIR / ".cache" / "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT_DIR / ".cache" / "triton"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_file, stream_with_context
from werkzeug.serving import WSGIRequestHandler
from tts_server.inference_worker import InferenceBusy, InferenceWorker, PreparationCancelled, PreparationRegistry

INFERENCE_WORKER = InferenceWorker()
PREPARATIONS = PreparationRegistry()


def generation_admitted(view=None, *, idle_only=False):
    """Reject overload before parsing uploads; retain admission through inference."""
    if view is None:
        return lambda wrapped: generation_admitted(wrapped, idle_only=idle_only)
    @wraps(view)
    def admitted(*args, **kwargs):
        try:
            with INFERENCE_WORKER.reserve(idle_only=idle_only) as admission:
                return view(admission, *args, **kwargs)
        except InferenceBusy as exc:
            response = jsonify({
                "ok": False,
                "error": str(exc),
                "code": "tts_busy",
                "retry_after": exc.retry_after,
            })
            response.status_code = 503
            response.headers["Retry-After"] = str(exc.retry_after)
            response.headers["Cache-Control"] = "no-store"
            return response
    return admitted


class LowLatencyRequestHandler(WSGIRequestHandler):
    # PCM and HTTP headers are small separate writes; do not wait for delayed ACKs.
    disable_nagle_algorithm = True

from tts_server.audio_trim import (
    first_rms_onset_sample as shared_first_rms_onset_sample,
    trim_audio_edges,
)

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

SETTINGS_DIR = ROOT_DIR / "settings"
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR = SETTINGS_DIR / "voxcpm_generations"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = SETTINGS_DIR / "voxcpm_server.json"
TEMPLATE_DIR = ROOT_DIR / "backend" / "voxcpm_web" / "templates"
STATIC_DIR = ROOT_DIR / "backend" / "voxcpm_web" / "static"
DEFAULT_LORA_SEARCH_PATHS = [
    str(ROOT_DIR / "loras" / "voxcpm2"),
]
LORA_WEIGHT_FILENAMES = (
    "lora_weights.safetensors",
    "lora_weights.ckpt",
    "lora_weights.pth",
    "lora_weights.bin",
)
LORA_IMPORT_FILENAMES = set(LORA_WEIGHT_FILENAMES) | {"lora_config.json", "lora_manifest.json"}
SAFE_LORA_NAME = re.compile(r"[^A-Za-z0-9._-]+")

SUPPORTED_LANGUAGES = [
    "Arabic",
    "Burmese",
    "Chinese",
    "Danish",
    "Dutch",
    "English",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Indonesian",
    "Italian",
    "Japanese",
    "Khmer",
    "Korean",
    "Lao",
    "Malay",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tagalog",
    "Thai",
    "Turkish",
    "Vietnamese",
]

SUPPORTED_DIALECTS = [
    "Sichuanese",
    "Cantonese",
    "Wu",
    "Dongbei",
    "Henan",
    "Shaanxi",
    "Shandong",
    "Tianjin",
    "Minnan",
]

THEME_NAMES = [
    "candy-nebula",
    "ember-glass",
    "acid-pop",
    "lagoon-burst",
    "rose-circuit",
    "aurora-pulse",
    "midnight-gold",
    "mint-fission",
    "tangerine-juice",
    "light-sherbet",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "model_id": "openbmb/VoxCPM2",
    "engine_preference": "standard",
    "default_mode": "ultimate_clone",
    "default_cfg_value": 2.0,
    "default_inference_timesteps": 8,     # what UltraTranslate requests; the startup warm-up captures this
    "default_max_generate_length": 2000,
    "default_temperature": 1.0,
    "default_speed": 1.0,
    "default_lora_path": "",
    "lora_search_paths": DEFAULT_LORA_SEARCH_PATHS,
    "default_streaming": True,
    "default_streaming_prefix_len": 2,
    "default_normalize": False,
    "default_use_memory_cache": "on",
    "default_denoise_reference": False,
    "default_trim_reference_silence": False,
    "default_reference_trim_mode": "full",
    "default_reference_lead_threshold_db": -40.0,
    "default_reference_lead_window_ms": 10,
    "default_reference_lead_preroll_ms": 20,
    "default_trim_output_leading_silence": True,
    "default_output_lead_threshold_db": -45.0,
    "default_output_lead_window_ms": 5,
    "default_output_lead_preroll_ms": 20,
    "default_output_lead_max_gate_ms": 1500,
    "stream_sample_rate": 48000,
    "optimize": True,
    "load_denoiser": False,
    "gpu_index": 0,
    "gpu_memory_utilization": 0.9,
    "nano_vllm_max_num_batched_tokens": 1536,
    "nano_vllm_max_num_seqs": 1,
    "nano_vllm_max_model_len": 1536,
    "nano_vllm_enforce_eager": False,
    "nano_vllm_max_reference_seconds": 4.0,
    "max_cache_entries": 24,
    "fastpath": True,
    "fastpath_prefix": True,
    "theme": "candy-nebula",
    "server_name": "UltraTranslate VoxCPM2 Server",
}

OPENAPI_SCHEMA = {
    "openapi": "3.1.0",
    "info": {
        "title": "UltraTranslate VoxCPM2 Server",
        "version": "2.0.0",
    },
    "paths": {
        "/v1/tts": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/VoxCPMCompatRequest"}
                        }
                    }
                }
            }
        }
    },
    "components": {
        "schemas": {
            "VoxCPMCompatRequest": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "references": {"type": "array"},
                    "reference_id": {"type": ["string", "null"]},
                    "normalize": {"type": "boolean"},
                    "format": {"type": "string"},
                    "max_new_tokens": {"type": "integer"},
                    "chunk_length": {"type": "integer"},
                    "top_p": {"type": "number"},
                    "repetition_penalty": {"type": "number"},
                    "temperature": {"type": "number"},
                    "streaming": {"type": "boolean"},
                    "streaming_prefix_len": {"type": "integer"},
                    "use_memory_cache": {"type": ["string", "boolean"]},
                    "seed": {"type": ["integer", "null"]},
                    "speed": {"type": "number"},
                    "lora_enabled": {"type": "boolean"},
                    "lora_path": {"type": "string"},
                    "lora_name": {"type": "string"},
                    "mode": {"type": "string"},
                    "control": {"type": "string"},
                    "cfg_value": {"type": "number"},
                    "inference_timesteps": {"type": "integer"},
                    "denoise_reference": {"type": "boolean"},
                    "trim_reference_silence": {"type": "boolean"},
                    "reference_trim_mode": {"type": "string"},
                    "reference_lead_threshold_db": {"type": "number"},
                    "reference_lead_window_ms": {"type": "number"},
                    "reference_lead_preroll_ms": {"type": "number"},
                    "trim_output_leading_silence": {"type": "boolean"},
                    "output_lead_threshold_db": {"type": "number"},
                    "output_lead_window_ms": {"type": "number"},
                    "output_lead_preroll_ms": {"type": "number"},
                    "output_lead_max_gate_ms": {"type": "number"},
                },
            }
        }
    },
}

logger = logging.getLogger("voxcpm_server")
_TORCH_RUNTIME_CONFIGURED = False


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_torch_runtime() -> None:
    global _TORCH_RUNTIME_CONFIGURED
    if _TORCH_RUNTIME_CONFIGURED:
        return
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    try:
        import torch

        torch.set_num_threads(max(1, int(os.environ.get('VOXCPM_CPU_THREADS', '4'))))
        torch.set_num_interop_threads(max(1, int(os.environ.get('VOXCPM_INTEROP_THREADS', '1'))))

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    except Exception as exc:  # pragma: no cover
        logger.debug("Torch runtime tuning skipped: %s", exc)
    _TORCH_RUNTIME_CONFIGURED = True


CUDA_CACHE_CEILING_BYTES = int(float(os.environ.get("VOXCPM_CUDA_CACHE_CEILING_GB", "6.4")) * 1024**3)


def release_cuda_cache_if_large() -> None:
    """Return cached-but-free VRAM to the shared GPU (ASR runs beside us) after a
    stream, only when the reserve has grown past the ceiling and nothing is queued."""
    try:
        import torch

        if (torch.cuda.is_available() and torch.cuda.memory_reserved() > CUDA_CACHE_CEILING_BYTES
                and not INFERENCE_WORKER.speech_waiting()):
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def mark_torch_step_begin() -> None:
    try:
        import torch

        compiler = getattr(torch, "compiler", None)
        if compiler is None:
            return
        marker = getattr(compiler, "cudagraph_mark_step_begin", None)
        if marker is None:
            marker = getattr(compiler, "mark_step_begin", None)
        if marker is not None:
            marker()
    except Exception:  # pragma: no cover
        return


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        resolved = int(value)
    except Exception:
        resolved = int(default)
    return max(minimum, min(maximum, resolved))


def clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        resolved = float(value)
    except Exception:
        resolved = float(default)
    return max(minimum, min(maximum, resolved))


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "y"}:
        return True
    if normalized in {"0", "false", "no", "off", "n"}:
        return False
    return default


def cache_flag(value: Any, default: str = "on") -> bool:
    if isinstance(value, bool):
        return value
    return str(value if value is not None else default).strip().lower() in {"1", "true", "yes", "on"}


def normalize_reference_trim_mode(value: Any, default: str = "full") -> str:
    normalized = str(value if value is not None else default).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"lead", "leading", "leading_only", "start", "front"}:
        return "leading"
    if normalized in {"tail", "trailing", "trailing_only", "ending", "ending_only", "end", "back"}:
        return "trailing"
    if normalized in {"full", "all", "both", "leading_trailing", "true", "on", "1", "yes"}:
        return "full"
    if normalized in {"off", "none", "raw", "false", "0", "no"}:
        return "off"
    return normalize_reference_trim_mode(default, "full") if normalized != str(default).strip().lower() else "full"


def _expand_lora_search_paths(config: Optional[dict[str, Any]] = None) -> list[Path]:
    paths: list[str] = []
    env_paths = os.environ.get("VOXCPM_LORA_PATHS", "")
    if env_paths:
        paths.extend([part for part in env_paths.split(os.pathsep) if part.strip()])
    raw_config_paths = (config or {}).get("lora_search_paths", DEFAULT_LORA_SEARCH_PATHS)
    if isinstance(raw_config_paths, str):
        paths.extend([part for part in raw_config_paths.split(os.pathsep) if part.strip()])
    elif isinstance(raw_config_paths, list):
        paths.extend([str(part) for part in raw_config_paths if str(part).strip()])
    paths.extend(DEFAULT_LORA_SEARCH_PATHS)

    resolved: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        try:
            path = Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()
        except Exception:
            continue
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            resolved.append(path)
    return resolved


def _contains_lora_weights(path: Path) -> bool:
    return path.is_dir() and any((path / filename).exists() for filename in LORA_WEIGHT_FILENAMES)


def _safe_lora_name(value: Any) -> str:
    cleaned = SAFE_LORA_NAME.sub("_", str(value or "").strip()).strip("._-")
    if not cleaned:
        raise ValueError("LoRA name is required")
    return cleaned[:96]


def _lora_import_root(config: Optional[dict[str, Any]] = None) -> Path:
    override = os.environ.get("VOXCPM_LORA_IMPORT_ROOT", "").strip()
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override))).resolve()
    paths = _expand_lora_search_paths(config)
    for path in paths:
        normalized = str(path).replace("\\", "/").rstrip("/")
        if normalized.endswith("/loras/voxcpm2"):
            return path
    if paths:
        return paths[0]
    return ROOT_DIR / "loras" / "voxcpm2"


def _candidate_lora_checkpoint(path: Path) -> Optional[Path]:
    if path.is_file() and path.name in LORA_WEIGHT_FILENAMES:
        return path.parent.resolve()
    if _contains_lora_weights(path):
        return path.resolve()
    for relative in (
        Path("latest"),
        Path("train") / "checkpoints" / "lora" / "latest",
        Path("checkpoints") / "lora" / "latest",
    ):
        candidate = path / relative
        if _contains_lora_weights(candidate):
            return candidate.resolve()
    return None


def _lora_display_name(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    try:
        train_index = parts.index("train")
        if train_index > 0:
            return path.parts[train_index - 1]
    except ValueError:
        pass
    if path.name.lower() == "latest" and len(path.parents) >= 4:
        return path.parents[3].name
    return path.name


def list_lora_checkpoints(config: Optional[dict[str, Any]] = None, *, limit: int = 200) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for root in _expand_lora_search_paths(config):
        if not root.exists():
            continue
        direct = _candidate_lora_checkpoint(root)
        candidates: list[Path] = [direct] if direct is not None else []
        if root.is_dir():
            try:
                for marker in root.rglob(LORA_WEIGHT_FILENAMES[0]):
                    candidates.append(marker.parent.resolve())
                    if len(candidates) >= limit:
                        break
            except Exception as exc:
                state.log(f"LoRA scan skipped {root}: {exc}") if "state" in globals() else None
        for checkpoint in candidates:
            if checkpoint is None or not _contains_lora_weights(checkpoint):
                continue
            key = str(checkpoint).lower()
            if key in found:
                continue
            name = _lora_display_name(checkpoint)
            found[key] = {
                "name": name,
                "path": str(checkpoint),
                "weights": next((str((checkpoint / f).resolve()) for f in LORA_WEIGHT_FILENAMES if (checkpoint / f).exists()), ""),
            }
            if len(found) >= limit:
                break
    return sorted(found.values(), key=lambda item: (str(item.get("name", "")).lower(), str(item.get("path", "")).lower()))


def resolve_lora_path(raw_value: Any, config: dict[str, Any]) -> str:
    raw = str(raw_value or "").strip()
    if raw.lower() in {"", "none", "off", "false", "no", "base", "base_model"}:
        return ""
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if expanded.exists():
        checkpoint = _candidate_lora_checkpoint(expanded.resolve())
        if checkpoint:
            return str(checkpoint)
        raise ValueError(f"LoRA path does not contain VoxCPM2 LoRA weights: {expanded}")
    for item in list_lora_checkpoints(config):
        if raw.lower() in {str(item.get("name", "")).lower(), str(item.get("path", "")).lower()}:
            return str(item["path"])
    if any(sep in raw for sep in ("/", "\\")):
        raise ValueError(f"LoRA path not found on this server: {raw}")
    raise ValueError(f"LoRA name not found on this server: {raw}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_preview(text: str, limit: int = 96) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def build_controlled_text(text: str, control: str) -> str:
    control = (control or "").strip()
    text = (text or "").strip()
    return f"({control}){text}" if control else text


def to_float_mono(audio: Any) -> np.ndarray:
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=0)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    arr = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


@contextmanager
def temporary_generation_seed(seed: Optional[int]) -> Iterator[None]:
    if seed is None:
        yield
        return
    seed_int = int(seed) & 0xFFFFFFFF
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch = None
    torch_state = None
    cuda_states = None
    try:
        import torch as torch_module

        torch = torch_module
        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
    except Exception:
        torch = None
    try:
        random.seed(seed_int)
        np.random.seed(seed_int)
        if torch is not None:
            torch.manual_seed(seed_int)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_int)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            try:
                torch.random.set_rng_state(torch_state)
                if cuda_states is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass


def speed_adjust(audio: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-6:
        return audio
    if speed <= 0:
        raise ValueError("speed must be greater than 0")
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    target_len = max(1, int(round(arr.size / speed)))
    if target_len == arr.size:
        return arr
    return np.interp(
        np.linspace(0.0, 1.0, target_len, dtype=np.float32),
        np.linspace(0.0, 1.0, arr.size, dtype=np.float32),
        arr,
    ).astype(np.float32)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    if target_rate <= 0 or source_rate <= 0:
        raise ValueError("sample rates must be positive")
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    if librosa is not None:
        return librosa.resample(arr, orig_sr=source_rate, target_sr=target_rate).astype(np.float32)
    target_len = max(1, int(round(arr.size * float(target_rate) / float(source_rate))))
    return np.interp(
        np.linspace(0.0, 1.0, target_len, dtype=np.float32),
        np.linspace(0.0, 1.0, arr.size, dtype=np.float32),
        arr,
    ).astype(np.float32)


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    if not audio_bytes:
        raise ValueError("reference audio is empty")
    if librosa is None:
        raise RuntimeError("librosa is required to decode reference audio")
    audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def encode_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), int(sample_rate), format="WAV")
    return buf.getvalue()


def trim_audio_bytes(audio_bytes: bytes) -> bytes:
    if librosa is None:
        return audio_bytes
    audio, sample_rate = decode_audio_bytes(audio_bytes)
    trimmed, _ = librosa.effects.trim(audio, top_db=32)
    if trimmed.size == 0:
        trimmed = audio
    return encode_wav_bytes(trimmed, sample_rate)


def clip_audio_bytes(audio_bytes: bytes, max_seconds: float) -> bytes:
    if max_seconds <= 0:
        return audio_bytes
    audio, sample_rate = decode_audio_bytes(audio_bytes)
    max_samples = max(1, int(round(float(sample_rate) * float(max_seconds))))
    if audio.shape[0] <= max_samples:
        return audio_bytes
    return encode_wav_bytes(audio[:max_samples], sample_rate)


def first_rms_onset_sample(audio: np.ndarray, sample_rate: int, threshold_dbfs: float, window_ms: float) -> Optional[int]:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    window = max(1, int(round(float(sample_rate) * float(window_ms) / 1000.0)))
    if arr.size < window:
        return None
    threshold_power = float(10.0 ** (float(threshold_dbfs) / 10.0))
    squared = arr.astype(np.float64) ** 2
    cumsum = np.concatenate(([0.0], np.cumsum(squared)))
    window_power = (cumsum[window:] - cumsum[:-window]) / float(window)
    hits = np.flatnonzero(window_power >= threshold_power)
    if hits.size == 0:
        return None
    return int(hits[0])


def trim_leading_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float,
    window_ms: float,
    preroll_ms: float,
) -> tuple[np.ndarray, int]:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    onset = first_rms_onset_sample(arr, sample_rate, threshold_dbfs, window_ms)
    if onset is None:
        return arr, 0
    preroll = max(0, int(round(float(sample_rate) * float(preroll_ms) / 1000.0)))
    start = max(0, onset - preroll)
    return arr[start:], start


def trim_reference_audio_bytes(
    audio_bytes: bytes,
    mode: str,
    *,
    threshold_dbfs: float = -40.0,
    window_ms: float = 10.0,
    preroll_ms: float = 20.0,
) -> bytes:
    resolved_mode = normalize_reference_trim_mode(mode, "off")
    if resolved_mode == "off":
        return audio_bytes
    audio, sample_rate = decode_audio_bytes(audio_bytes)
    trimmed, _lead_dropped, _tail_dropped, _lead_onset, _tail_onset = trim_audio_edges(
        audio,
        sample_rate,
        trim_leading=resolved_mode in {"leading", "full"},
        trim_trailing=resolved_mode in {"trailing", "full"},
        threshold_dbfs=threshold_dbfs,
        window_ms=window_ms,
        preroll_ms=preroll_ms,
    )
    return encode_wav_bytes(trimmed, sample_rate)


def iter_lead_trimmed_audio(
    chunks: Iterator[np.ndarray],
    sample_rate: int,
    *,
    threshold_dbfs: float,
    window_ms: float,
    preroll_ms: float,
    max_gate_ms: float,
) -> Iterator[np.ndarray]:
    buffered: list[np.ndarray] = []
    buffered_samples = 0
    max_gate_samples = max(1, int(round(float(sample_rate) * float(max_gate_ms) / 1000.0)))
    gate_open = False

    for chunk in chunks:
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        if gate_open:
            yield arr
            continue

        buffered.append(arr)
        buffered_samples += int(arr.size)
        joined = np.concatenate(buffered).astype(np.float32) if len(buffered) > 1 else buffered[0]
        onset = first_rms_onset_sample(joined, sample_rate, threshold_dbfs, window_ms)
        if onset is not None:
            preroll = max(0, int(round(float(sample_rate) * float(preroll_ms) / 1000.0)))
            start = max(0, onset - preroll)
            gate_open = True
            if start < joined.size:
                yield joined[start:]
            buffered.clear()
            continue

        if buffered_samples >= max_gate_samples:
            gate_open = True
            yield joined
            buffered.clear()

    if not gate_open and buffered:
        yield np.concatenate(buffered).astype(np.float32) if len(buffered) > 1 else buffered[0]


@dataclass
class GenerationRequest:
    text: str
    mode: str
    control: str
    normalize: bool
    streaming: bool
    speed: float
    cfg_value: float
    inference_timesteps: int
    max_generate_length: int
    temperature: float
    use_memory_cache: bool
    denoise_reference: bool
    trim_reference_silence: bool
    reference_trim_mode: str
    reference_lead_threshold_db: float
    reference_lead_window_ms: float
    reference_lead_preroll_ms: float
    trim_output_leading_silence: bool
    output_lead_threshold_db: float
    output_lead_window_ms: float
    output_lead_preroll_ms: float
    output_lead_max_gate_ms: float
    response_format: str
    streaming_prefix_len: int
    lora_path: str = ""
    lora_name: str = ""
    reference_id: Optional[str] = None
    reference_audio_bytes: Optional[bytes] = None
    reference_transcript: str = ""
    source: str = "api"
    seed: Optional[int] = None
    fastpath: bool = True
    fastpath_prefix: bool = True


@dataclass
class ExecutionPlan:
    request: GenerationRequest
    engine_kind: str
    sample_rate: int
    cache_hit: bool
    reference_id: Optional[str]
    chunk_iterator_factory: Callable[[], Iterator[np.ndarray]]


@dataclass
class CacheEntry:
    cache_key: str
    reference_id: str
    engine_kind: str
    mode: str
    payload: Any
    digest: str
    transcript: str
    created_at: str
    updated_at: str
    hits: int = 0


@dataclass
class JobRecord:
    job_id: str
    source: str
    mode: str
    text_preview: str
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None
    engine: str = ""
    cache_hit: bool = False
    reference_id: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        return asdict(self)


class ServerState:
    def __init__(self) -> None:
        self._config_lock = threading.RLock()
        self._jobs_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._log_lock = threading.RLock()
        self.config = self._load_config()
        self.jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cache_aliases: dict[str, str] = {}
        self.logs: deque[str] = deque(maxlen=400)
        self.generation_lock = threading.Lock()
        self.engine_manager = EngineManager(self)

    def _load_config(self) -> dict[str, Any]:
        data = dict(DEFAULT_CONFIG)
        if SETTINGS_PATH.exists():
            try:
                loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data.update(loaded)
            except Exception as exc:  # pragma: no cover
                logger.warning("Config load failed: %s", exc)
        return data

    def save_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self._config_lock:
            changed = any(self.config.get(key) != value for key, value in updates.items())
            self.config.update(updates)
            SETTINGS_PATH.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
            if changed:
                self.engine_manager.mark_dirty()
            return dict(self.config)

    def get_config(self) -> dict[str, Any]:
        with self._config_lock:
            return dict(self.config)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        with self._log_lock:
            self.logs.appendleft(line)
        logger.info(message)

    def get_logs(self) -> list[str]:
        with self._log_lock:
            return list(self.logs)

    def create_job(self, source: str, mode: str, text: str) -> JobRecord:
        job = JobRecord(
            job_id=uuid.uuid4().hex,
            source=source,
            mode=mode,
            text_preview=text_preview(text),
        )
        with self._jobs_lock:
            self.jobs[job.job_id] = job
            while len(self.jobs) > 40:
                self.jobs.popitem(last=False)
        return job

    def update_job(self, job_id: str, **changes: Any) -> None:
        with self._jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = now_iso()

    def complete_job(self, job_id: str, **changes: Any) -> None:
        changes.setdefault("status", "completed")
        changes.setdefault("progress", 100)
        changes.setdefault("message", "Completed")
        changes.setdefault("completed_at", now_iso())
        self.update_job(job_id, **changes)

    def fail_job(self, job_id: str, error: str) -> None:
        self.update_job(
            job_id,
            status="error",
            progress=100,
            message="Generation failed",
            error=error,
            completed_at=now_iso(),
        )

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._jobs_lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._jobs_lock:
            return [job.to_public() for job in reversed(list(self.jobs.values()))]

    def clear_cache(self) -> None:
        with self._cache_lock:
            self.cache.clear()
            self.cache_aliases.clear()

    def get_cache_entry(self, cache_key: Optional[str] = None, reference_id: Optional[str] = None) -> Optional[CacheEntry]:
        with self._cache_lock:
            resolved_key = cache_key
            if reference_id:
                resolved_key = self.cache_aliases.get(reference_id)
            if not resolved_key:
                return None
            entry = self.cache.get(resolved_key)
            if entry is not None:
                entry.hits += 1
                entry.updated_at = now_iso()
            return entry

    def put_cache_entry(self, entry: CacheEntry, *, allow_eviction: bool = True) -> None:
        with self._cache_lock:
            max_entries = clamp_int(self.config.get("max_cache_entries"), 24, 4, 128)
            if not allow_eviction and entry.cache_key not in self.cache and len(self.cache) >= max_entries:
                raise InferenceBusy("Reference cache is full; preparation will not evict existing voices")
            self.cache[entry.cache_key] = entry
            self.cache_aliases[entry.reference_id] = entry.cache_key
            self.cache.move_to_end(entry.cache_key)
            while len(self.cache) > max_entries:
                _, old_entry = self.cache.popitem(last=False)
                self.cache_aliases.pop(old_entry.reference_id, None)

    def check_preparation_cache_capacity(self) -> None:
        with self._cache_lock:
            if len(self.cache) >= clamp_int(self.config.get("max_cache_entries"), 24, 4, 128):
                raise InferenceBusy("Reference cache is full; preparation will not evict existing voices")

    def cache_snapshot(self) -> list[dict[str, Any]]:
        with self._cache_lock:
            return [
                {
                    "reference_id": entry.reference_id,
                    "mode": entry.mode,
                    "engine_kind": entry.engine_kind,
                    "digest": entry.digest[:16],
                    "transcript_preview": text_preview(entry.transcript, 42),
                    "hits": entry.hits,
                    "updated_at": entry.updated_at,
                }
                for entry in reversed(list(self.cache.values()))
            ]


class StandardEngine:
    def __init__(self, model_id: str, *, optimize: bool, load_denoiser: bool) -> None:
        configure_torch_runtime()

        self.kind = "standard"
        self.model_id = model_id
        self.optimize = bool(optimize)
        self.load_denoiser = bool(load_denoiser)
        self.lora_path = ""
        self._loaded_lora_path = ""
        self._lora_config_signature: Optional[tuple[Any, ...]] = None
        # Add adapters only when a request actually selects a LoRA checkpoint.
        # Zero-scaled LoRA layers still execute their two extra matrix products.
        self.model = self._load_model(lora_config=None)
        self.sample_rate = int(getattr(self.model.tts_model, "sample_rate", 48000))
        self.is_offloaded = False
        self.fast = None
        self._attach_fastpath()

    def _attach_fastpath(self) -> None:
        """CUDA-graph decode (same weights/sampling); stock path stays as fallback."""
        self.fast = None
        if not coerce_bool(os.environ.get("VOXCPM_FASTPATH", "1"), True):
            return
        tts = getattr(self.model, "tts_model", None)
        if tts is None or getattr(tts, "lora_config", None) is not None:
            return  # LoRA layers toggle in Python; captured graphs would ignore that.
        try:
            from tts_server.voxcpm_fastpath import VoxCPM2FastPath

            self.fast = VoxCPM2FastPath(tts)
            logger.info("VoxCPM2 fast path attached (flash=%s, folded=%d)", self.fast.use_flash, self.fast.folded)
        except Exception as exc:  # pragma: no cover - depends on GPU runtime
            logger.warning("VoxCPM2 fast path unavailable: %s", exc)
            self.fast = None

    def warm_fastpath(self, inference_timesteps: int, cfg_value: float) -> None:
        if self.fast is None:
            return
        started = time.perf_counter()
        try:
            self.fast.warm(inference_timesteps, cfg_value)
            logger.info("Fast path graphs captured in %.2fs (compile=%s)", time.perf_counter() - started,
                        self.fast.compile_mode)
            return
        except Exception as exc:  # pragma: no cover
            logger.warning("Fast path capture failed (compile=%s): %s", getattr(self.fast, "compile_mode", None), exc)
        # Retry with eager kernels inside the graphs before giving up on the fast path.
        try:
            from tts_server.voxcpm_fastpath import VoxCPM2FastPath

            self.fast = None
            self.fast = VoxCPM2FastPath(self.model.tts_model, compile_mode="off")
            self.fast.warm(inference_timesteps, cfg_value)
            logger.info("Fast path (uncompiled) captured in %.2fs", time.perf_counter() - started)
        except Exception as exc:  # pragma: no cover
            logger.warning("Fast path unavailable; using stock path: %s", exc)
            self.fast = None

    def prepare_fastpath_prefix(self, prompt_cache: Any) -> None:
        if self.fast is not None and self.fast.use_prefix and prompt_cache is not None:
            try:
                self.fast.prepare(prompt_cache)
            except Exception as exc:  # pragma: no cover
                logger.warning("Fast path voice prefix failed (stock prefill will be used): %s", exc)

    def _default_lora_config(self) -> Any:
        try:
            from voxcpm.model.voxcpm2 import LoRAConfig
        except Exception:
            from voxcpm.model.voxcpm import LoRAConfig
        return LoRAConfig(
            enable_lm=True,
            enable_dit=True,
            enable_proj=False,
            r=32,
            alpha=32,
            dropout=0.0,
            target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
            target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
        )

    def _lora_config_from_checkpoint(self, lora_path: str) -> Any:
        config_file = Path(lora_path) / "lora_config.json"
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
                cfg = data.get("lora_config") if isinstance(data, dict) else None
                if isinstance(cfg, dict) and cfg:
                    try:
                        from voxcpm.model.voxcpm2 import LoRAConfig
                    except Exception:
                        from voxcpm.model.voxcpm import LoRAConfig
                    return LoRAConfig(**cfg)
            except Exception as exc:
                state.log(f"Failed to read LoRA config {config_file}: {exc}") if "state" in globals() else None
        return self._default_lora_config()

    def _lora_signature(self, lora_config: Any) -> tuple[Any, ...]:
        def _field(name: str, default: Any = None) -> Any:
            value = getattr(lora_config, name, default)
            return tuple(value) if isinstance(value, list) else value
        return (
            bool(_field("enable_lm", False)),
            bool(_field("enable_dit", False)),
            bool(_field("enable_proj", False)),
            int(_field("r", 32)),
            int(_field("alpha", 32)),
            tuple(_field("target_modules_lm", []) or []),
            tuple(_field("target_modules_dit", []) or []),
            tuple(_field("target_proj_modules", []) or []),
        )

    def _load_model(self, *, lora_config: Any, lora_weights_path: Optional[str] = None) -> Any:
        from voxcpm import VoxCPM

        model = VoxCPM.from_pretrained(
            hf_model_id=self.model_id,
            load_denoiser=self.load_denoiser,
            optimize=self.optimize,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
        )
        self._lora_config_signature = self._lora_signature(lora_config)
        return model

    def _replace_model_for_lora(self, lora_path: str, lora_config: Any) -> None:
        old_model = getattr(self, "model", None)
        self.model = None
        if old_model is not None:
            try:
                del old_model
            except Exception:
                pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass
        self.fast = None  # graphs referenced the old model
        self.model = self._load_model(lora_config=lora_config, lora_weights_path=lora_path)
        self.sample_rate = int(getattr(self.model.tts_model, "sample_rate", 48000))
        self._loaded_lora_path = lora_path
        self.lora_path = lora_path

    def ensure_lora(self, lora_path: str) -> None:
        requested = str(lora_path or "").strip()
        if requested == self.lora_path:
            return
        if not requested:
            if self._loaded_lora_path:
                if not hasattr(self.model, "set_lora_enabled"):
                    raise RuntimeError("Loaded VoxCPM2 model does not support disabling LoRA")
                self.model.set_lora_enabled(False)
            self.lora_path = ""
            return
        if requested == self._loaded_lora_path:
            if not hasattr(self.model, "set_lora_enabled"):
                raise RuntimeError("Loaded VoxCPM2 model does not support enabling LoRA")
            self.model.set_lora_enabled(True)
            self.lora_path = requested
            return
        requested_config = self._lora_config_from_checkpoint(requested)
        requested_signature = self._lora_signature(requested_config)
        if requested_signature != self._lora_config_signature:
            self._replace_model_for_lora(requested, requested_config)
            return
        if self._loaded_lora_path:
            if not hasattr(self.model, "unload_lora"):
                raise RuntimeError("Loaded VoxCPM2 model does not support switching LoRA")
            self.model.unload_lora()
            self._loaded_lora_path = ""
            self.lora_path = ""
        if not hasattr(self.model, "load_lora"):
            raise RuntimeError("Loaded VoxCPM2 model does not support LoRA hot-swap")
        self.model.load_lora(requested)
        self._loaded_lora_path = requested
        self.lora_path = requested

    def stop(self) -> None:
        model = getattr(self, "model", None)
        self.model = None
        if model is not None:
            try:
                del model
            except Exception:
                pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def offload(self) -> None:
        self.fast = None  # CUDA graphs hold device addresses that offloading invalidates
        if not getattr(self, "is_offloaded", False):
            if hasattr(self, "model") and self.model is not None:
                if hasattr(self.model, "tts_model") and self.model.tts_model is not None:
                    self.model.tts_model.to("cpu")
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            self.is_offloaded = True

    def onload(self) -> None:
        if getattr(self, "is_offloaded", False):
            if hasattr(self, "model") and self.model is not None:
                if hasattr(self.model, "tts_model") and self.model.tts_model is not None:
                    self.model.tts_model.to("cuda")
            self.is_offloaded = False
            self._attach_fastpath()  # graphs are recaptured lazily on first use

    def normalize_text(self, text: str) -> str:
        if self.model.text_normalizer is None:
            from voxcpm.utils.text_normalize import TextNormalizer

            self.model.text_normalizer = TextNormalizer()
        return self.model.text_normalizer.normalize(text)

    def build_prompt_cache(
        self,
        *,
        reference_audio_bytes: bytes,
        prompt_text: str,
        mode: str,
        reference_trim_mode: str,
        reference_lead_threshold_db: float,
        reference_lead_window_ms: float,
        reference_lead_preroll_ms: float,
    ) -> Any:
        if reference_trim_mode != "off":
            reference_audio_bytes = trim_reference_audio_bytes(
                reference_audio_bytes,
                reference_trim_mode,
                threshold_dbfs=reference_lead_threshold_db,
                window_ms=reference_lead_window_ms,
                preroll_ms=reference_lead_preroll_ms,
            )
        reference_audio_path = _canonical_audio_temp_path(reference_audio_bytes)
        prompt_audio_path = reference_audio_path if mode == "ultimate_clone" else None
        trim_silence_vad = False

        def build():
            if mode == "clone":
                return self.model.tts_model.build_prompt_cache(
                    reference_wav_path=reference_audio_path,
                    trim_silence_vad=trim_silence_vad,
                )
            return self.model.tts_model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_audio_path,
                reference_wav_path=reference_audio_path,
                trim_silence_vad=trim_silence_vad,
            )

        try:
            try:
                return build()
            except Exception as exc:
                # Long references need a few hundred MB of VAE activations on a GPU shared
                # with ASR; hand cached blocks back to the driver and retry once.
                if "out of memory" not in str(exc).lower():
                    raise
                import torch

                torch.cuda.empty_cache()
                logger.warning("Reference encode hit CUDA OOM; retrying after releasing cache")
                return build()
        finally:
            Path(reference_audio_path).unlink(missing_ok=True)

    def generate_iterator(self, plan: ExecutionPlan) -> Iterator[np.ndarray]:
        req = plan.request
        mark_torch_step_begin()
        target_text = req.text
        if req.normalize:
            target_text = self.normalize_text(target_text)
        payload = getattr(plan, "_cached_payload", None)
        if isinstance(payload, dict):
            prompt_cache = payload.get("prompt_cache")
            if prompt_cache is not None and self.fast is not None and req.fastpath and not self.lora_path:
                fast_iter = None
                try:
                    self.fast.use_prefix = bool(req.fastpath_prefix)
                    fast_iter = self.fast.generate(
                        target_text, prompt_cache,
                        max_len=req.max_generate_length,
                        inference_timesteps=req.inference_timesteps,
                        cfg_value=req.cfg_value,
                        streaming_prefix_len=req.streaming_prefix_len,
                    )
                    first = next(fast_iter)
                except StopIteration:
                    return
                except Exception as exc:
                    # Nothing has been sent yet: answer this request on the stock path.
                    logger.warning("Fast path failed before audio; stock fallback: %s", exc)
                    self._fast_failures = getattr(self, "_fast_failures", 0) + 1
                    if self._fast_failures >= 2:
                        # e.g. no Triton on Windows: stop retrying the compile on every line (it costs seconds)
                        logger.warning("Fast path disabled after repeated failures (install triton-windows to use it)")
                        self.fast = None
                    if fast_iter is not None:
                        fast_iter.close()
                    fast_iter = None
                if fast_iter is not None:
                    try:
                        yield first.numpy()
                        for chunk in fast_iter:
                            yield chunk.numpy()
                    finally:
                        fast_iter.close()  # releases the fast-path lock on disconnect too
                    return
            if prompt_cache is not None:
                iterator = self.model.tts_model.generate_with_prompt_cache_streaming(
                    target_text=target_text,
                    prompt_cache=prompt_cache,
                    cfg_value=req.cfg_value,
                    inference_timesteps=req.inference_timesteps,
                    max_len=req.max_generate_length,
                    streaming_prefix_len=req.streaming_prefix_len,
                )
                for chunk, *_ in iterator:
                    yield to_float_mono(chunk)
                return
        reference_audio_bytes = req.reference_audio_bytes
        if reference_audio_bytes is not None and req.reference_trim_mode != "off":
            reference_audio_bytes = trim_reference_audio_bytes(
                reference_audio_bytes,
                req.reference_trim_mode,
                threshold_dbfs=req.reference_lead_threshold_db,
                window_ms=req.reference_lead_window_ms,
                preroll_ms=req.reference_lead_preroll_ms,
            )
        reference_audio_path = _canonical_audio_temp_path(reference_audio_bytes) if reference_audio_bytes else None
        prompt_audio_path = reference_audio_path if req.mode == "ultimate_clone" else None
        try:
            iterator = self.model.generate_streaming(
                text=target_text,
                prompt_wav_path=prompt_audio_path,
                prompt_text=req.reference_transcript if req.mode == "ultimate_clone" else None,
                reference_wav_path=reference_audio_path,
                cfg_value=req.cfg_value,
                inference_timesteps=req.inference_timesteps,
                max_len=req.max_generate_length,
                normalize=False,
                denoise=req.denoise_reference,
            )
            for chunk in iterator:
                yield to_float_mono(chunk)
        finally:
            if reference_audio_path:
                Path(reference_audio_path).unlink(missing_ok=True)


class NanoVLLMEngine:
    def __init__(
        self,
        model_id: str,
        *,
        inference_timesteps: int,
        gpu_index: int,
        gpu_memory_utilization: float,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_model_len: int,
        enforce_eager: bool,
    ) -> None:
        configure_torch_runtime()
        from nanovllm_voxcpm import VoxCPM
        from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

        self.kind = "nano_vllm"
        self.model_id = model_id
        lora_config = LoRAConfig()
        self.server = VoxCPM.from_pretrained(
            model=model_id,
            inference_timesteps=inference_timesteps,
            devices=[gpu_index],
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            lora_config=lora_config,
        )
        try:
            self.server.set_lora_enabled(False)
        except Exception:
            pass
        self.lora_path = ""
        self._loaded_lora_path = ""
        model_info = self.server.get_model_info()
        self.sample_rate = int(model_info.get("output_sample_rate") or model_info.get("sample_rate") or 48000)
        self.encoder_sample_rate = int(model_info.get("encoder_sample_rate") or 16000)

    def ensure_lora(self, lora_path: str) -> None:
        requested = str(lora_path or "").strip()
        if requested == self.lora_path:
            return
        if not requested:
            if self._loaded_lora_path:
                self.server.set_lora_enabled(False)
            self.lora_path = ""
            return
        if requested == self._loaded_lora_path:
            self.server.set_lora_enabled(True)
            self.lora_path = requested
            return
        self.server.load_lora(requested)
        self.server.set_lora_enabled(True)
        self._loaded_lora_path = requested
        self.lora_path = requested

    def stop(self) -> None:
        self.server.stop()

    def encode_latents(self, audio_bytes: bytes) -> bytes:
        return self.server.encode_latents(audio_bytes, "wav")

    def generate_iterator(self, plan: ExecutionPlan) -> Iterator[np.ndarray]:
        req = plan.request
        mark_torch_step_begin()
        payload = getattr(plan, "_cached_payload", None)
        prompt_latents = None
        ref_audio_latents = None
        prompt_text = ""
        if isinstance(payload, dict):
            prompt_latents = payload.get("prompt_latents")
            ref_audio_latents = payload.get("ref_audio_latents")
            prompt_text = payload.get("prompt_text", "")
        
        if "state" in globals():
            st = globals()["state"]
            st.log(f"DEBUG NANO_VLLM: target_text='{req.text}'")
            st.log(f"DEBUG NANO_VLLM: payload is dict={isinstance(payload, dict)}")
            if isinstance(payload, dict):
                st.log(f"DEBUG NANO_VLLM: payload keys={list(payload.keys())}")
                plat = payload.get("prompt_latents")
                rlat = payload.get("ref_audio_latents")
                st.log(f"DEBUG NANO_VLLM: prompt_latents type={type(plat)} size={len(plat) if plat is not None else 'None'}")
                st.log(f"DEBUG NANO_VLLM: ref_audio_latents type={type(rlat)} size={len(rlat) if rlat is not None else 'None'}")
                p_text = payload.get("prompt_text")
                st.log(f"DEBUG NANO_VLLM: prompt_text='{p_text}'")
            else:
                st.log("DEBUG NANO_VLLM: payload is NOT dict")

        for chunk in self.server.generate(
            target_text=req.text,
            prompt_latents=prompt_latents,
            prompt_text=prompt_text,
            max_generate_length=req.max_generate_length,
            temperature=req.temperature,
            cfg_value=req.cfg_value,
            ref_audio_latents=ref_audio_latents,
        ):
            yield to_float_mono(chunk)


class EngineManager:
    def __init__(self, state: ServerState) -> None:
        self.state = state
        self._lock = threading.RLock()
        self._engines: dict[tuple[Any, ...], Any] = {}
        self._engine_key: Optional[tuple[Any, ...]] = None
        self._dirty = False

    @property
    def _engine(self) -> Any:
        return self._engines.get(self._engine_key)

    def mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active_engine = self._engine
            if active_engine is None:
                return {"loaded": False}
            return {
                "loaded": True,
                "engine_kind": active_engine.kind,
                "model_id": getattr(active_engine, "model_id", ""),
                "lora_path": getattr(active_engine, "lora_path", ""),
                "sample_rate": getattr(active_engine, "sample_rate", 0),
            }

    def _build_key(self, config: dict[str, Any], *, force_standard: bool, inference_timesteps: int, lora_path: str = "") -> tuple[Any, ...]:
        preference = str(config.get("engine_preference", "auto")).strip().lower()
        if force_standard:
            preference = "standard"
        return (
            preference,
            str(config.get("model_id", DEFAULT_CONFIG["model_id"])),
            bool(config.get("optimize", True)),
            bool(config.get("load_denoiser", False)),
            clamp_int(config.get("gpu_index"), 0, 0, 16),
            clamp_float(config.get("gpu_memory_utilization"), 0.9, 0.5, 0.99),
            int(inference_timesteps) if preference in {"auto", "nano_vllm"} else None,
            clamp_int(config.get("nano_vllm_max_num_batched_tokens"), 1536, 256, 16384)
            if preference in {"auto", "nano_vllm"}
            else None,
            clamp_int(config.get("nano_vllm_max_num_seqs"), 1, 1, 128) if preference in {"auto", "nano_vllm"} else None,
            clamp_int(config.get("nano_vllm_max_model_len"), 1536, 512, 32768)
            if preference in {"auto", "nano_vllm"}
            else None,
            coerce_bool(config.get("nano_vllm_enforce_eager"), False) if preference in {"auto", "nano_vllm"} else None,
        )

    def reference_preparation_engine(self, req: GenerationRequest) -> Any:
        """Inspect existing engine only: never load, onload, switch or change LoRA."""
        config = self.state.get_config()
        key = self._build_key(config, force_standard=True, inference_timesteps=req.inference_timesteps)
        with self._lock:
            engine = self._engine
            if (self._dirty or self._engine_key != key or engine is None
                    or engine.kind != "standard" or getattr(engine, "is_offloaded", False)
                    or getattr(engine, "lora_path", "")):
                raise InferenceBusy("Current engine is not ready for background voice preparation")
            return engine

    def ensure_loaded(self, config: dict[str, Any], *, force_standard: bool, inference_timesteps: int, lora_path: str = "") -> Any:
        key = self._build_key(config, force_standard=force_standard, inference_timesteps=inference_timesteps, lora_path=lora_path)
        with self._lock:
            if self._dirty:
                for k, eng in list(self._engines.items()):
                    try:
                        eng.stop()
                    except Exception:
                        pass
                self._engines.clear()
                self._engine_key = None
                self._dirty = False
                self.state.clear_cache()

            if key in self._engines:
                target_engine = self._engines[key]
                if target_engine.kind == "standard" and getattr(target_engine, "is_offloaded", False):
                    # Stop any running nano_vllm engine to free VRAM
                    for k, eng in list(self._engines.items()):
                        if eng.kind == "nano_vllm":
                            self.state.log("Stopping Nano-vLLM engine to free VRAM for Standard engine.")
                            try:
                                eng.stop()
                            except Exception:
                                pass
                            self._engines.pop(k, None)
                    self.state.log("Onloading Standard engine back to GPU.")
                    target_engine.onload()
                self._engine_key = key
                return target_engine

            preference = str(config.get("engine_preference", "auto")).strip().lower()
            if force_standard:
                preference = "standard"
            target_kind = "nano_vllm" if preference in {"auto", "nano_vllm"} else "standard"

            if target_kind == "standard":
                # Stop any running nano_vllm engine to free VRAM
                for k, eng in list(self._engines.items()):
                    if eng.kind == "nano_vllm":
                        self.state.log("Stopping Nano-vLLM engine to free VRAM for Standard engine.")
                        try:
                            eng.stop()
                        except Exception:
                            pass
                        self._engines.pop(k, None)

            if target_kind == "nano_vllm":
                # Offload any running standard engine to CPU to free VRAM
                for k, eng in list(self._engines.items()):
                    if eng.kind == "standard":
                        self.state.log("Offloading Standard engine to CPU to free VRAM for Nano-vLLM.")
                        try:
                            eng.offload()
                        except Exception:
                            pass

            new_engine = self._load_engine(config, force_standard=force_standard, inference_timesteps=inference_timesteps, lora_path=lora_path)
            new_kind = new_engine.kind

            # Stop and remove any other engine of the SAME kind from the cache
            for k, eng in list(self._engines.items()):
                if eng.kind == new_kind and k != key:
                    self.state.log(f"Stopping and clearing older cached engine of kind: {new_kind} (key={k})")
                    try:
                        eng.stop()
                    except Exception:
                        pass
                    self._engines.pop(k, None)

            self._engines[key] = new_engine
            self._engine_key = key
            return new_engine

    def _load_engine(self, config: dict[str, Any], *, force_standard: bool, inference_timesteps: int, lora_path: str = "") -> Any:
        model_id = str(config.get("model_id") or DEFAULT_CONFIG["model_id"]).strip()
        optimize = bool(config.get("optimize", True))
        load_denoiser = bool(config.get("load_denoiser", False))
        gpu_index = clamp_int(config.get("gpu_index"), 0, 0, 16)
        gpu_memory_utilization = clamp_float(config.get("gpu_memory_utilization"), 0.9, 0.5, 0.99)
        nano_vllm_max_num_batched_tokens = clamp_int(config.get("nano_vllm_max_num_batched_tokens"), 1536, 256, 16384)
        nano_vllm_max_num_seqs = clamp_int(config.get("nano_vllm_max_num_seqs"), 1, 1, 128)
        nano_vllm_max_model_len = clamp_int(config.get("nano_vllm_max_model_len"), 1536, 512, 32768)
        nano_vllm_enforce_eager = coerce_bool(config.get("nano_vllm_enforce_eager"), False)
        preference = str(config.get("engine_preference", "auto")).strip().lower()
        if force_standard:
            preference = "standard"

        if preference in {"auto", "nano_vllm"}:
            try:
                self.state.log(
                    f"Loading Nano-vLLM VoxCPM2 engine: model={model_id} timesteps={inference_timesteps} gpu={gpu_index}"
                )
                return NanoVLLMEngine(
                    model_id,
                    inference_timesteps=inference_timesteps,
                    gpu_index=gpu_index,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_batched_tokens=nano_vllm_max_num_batched_tokens,
                    max_num_seqs=nano_vllm_max_num_seqs,
                    max_model_len=nano_vllm_max_model_len,
                    enforce_eager=nano_vllm_enforce_eager,
                )
            except Exception as exc:
                if preference == "nano_vllm":
                    raise
                self.state.log(f"Nano-vLLM unavailable, falling back to standard engine: {exc}")

        self.state.log(f"Loading standard VoxCPM2 engine: model={model_id}")
        return StandardEngine(model_id, optimize=optimize, load_denoiser=load_denoiser)


def _canonical_audio_temp_path(audio_bytes: bytes) -> str:
    audio, sample_rate = decode_audio_bytes(audio_bytes)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    handle.close()
    sf.write(handle.name, audio, sample_rate, format="WAV")
    return handle.name


state = ServerState()
app = Flask(__name__)


def parse_request_payload(payload: dict[str, Any], *, source: str) -> GenerationRequest:
    config = state.get_config()
    raw_text = str(payload.get("text") or "").strip()
    if not raw_text:
        raise ValueError("text is required")

    raw_references = payload.get("references") or []
    reference = raw_references[0] if isinstance(raw_references, list) and raw_references else {}
    encoded_audio = str(reference.get("audio") or "").strip() if isinstance(reference, dict) else ""
    reference_audio_bytes = None
    if encoded_audio:
        cleaned = encoded_audio.split("base64,", 1)[-1]
        reference_audio_bytes = base64.b64decode("".join(cleaned.split()))
    reference_id = str(payload.get("reference_id") or "").strip() or None

    control = str(payload.get("control") or payload.get("style_prompt") or payload.get("design_prompt") or "").strip()
    requested_mode = str(payload.get("mode") or config.get("default_mode") or "ultimate_clone").strip().lower()
    if requested_mode in {"auto", ""}:
        if reference_audio_bytes and str(reference.get("text") or "").strip():
            requested_mode = "ultimate_clone"
        elif reference_audio_bytes:
            requested_mode = "clone"
        elif control:
            requested_mode = "voice_design"
        else:
            requested_mode = "text"

    if requested_mode in {"ultimate", "ultimate_clone"}:
        requested_mode = "ultimate_clone"
    elif requested_mode in {"controllable_clone", "clone"}:
        requested_mode = "clone"
    elif requested_mode in {"design", "voice_design"}:
        requested_mode = "voice_design"
    else:
        requested_mode = "text"

    if requested_mode in {"clone", "ultimate_clone"} and reference_audio_bytes is None and reference_id is None:
        raise ValueError("reference audio or reference_id is required for clone modes")
    if requested_mode == "ultimate_clone" and reference_audio_bytes is not None and not str(reference.get("text") or "").strip():
        raise ValueError("ultimate cloning requires the reference transcript")

    normalize = coerce_bool(payload.get("normalize"), bool(config.get("default_normalize", False)))
    use_memory_cache = cache_flag(payload.get("use_memory_cache"), str(config.get("default_use_memory_cache", "on")))
    denoise_reference = coerce_bool(
        payload.get("denoise_reference"),
        bool(config.get("default_denoise_reference", False)),
    )
    trim_reference_silence = coerce_bool(
        payload.get("trim_reference_silence"),
        bool(config.get("default_trim_reference_silence", False)),
    )
    raw_reference_trim_mode = payload.get("reference_trim_mode")
    if raw_reference_trim_mode is not None:
        reference_trim_mode = normalize_reference_trim_mode(raw_reference_trim_mode, "full")
        trim_reference_silence = reference_trim_mode != "off"
    elif trim_reference_silence:
        reference_trim_mode = normalize_reference_trim_mode(config.get("default_reference_trim_mode"), "full")
        if reference_trim_mode == "off":
            trim_reference_silence = False
    else:
        reference_trim_mode = "off"
    reference_lead_threshold_db = clamp_float(
        payload.get("reference_lead_threshold_db"),
        config.get("default_reference_lead_threshold_db", -40.0),
        -90.0,
        -5.0,
    )
    reference_lead_window_ms = clamp_float(
        payload.get("reference_lead_window_ms"),
        config.get("default_reference_lead_window_ms", 10),
        1.0,
        100.0,
    )
    reference_lead_preroll_ms = clamp_float(
        payload.get("reference_lead_preroll_ms"),
        config.get("default_reference_lead_preroll_ms", 20),
        0.0,
        500.0,
    )
    trim_output_leading_silence = coerce_bool(
        payload.get("trim_output_leading_silence"),
        bool(config.get("default_trim_output_leading_silence", True)),
    )
    output_lead_threshold_db = clamp_float(
        payload.get("output_lead_threshold_db"),
        config.get("default_output_lead_threshold_db", -45.0),
        -90.0,
        -5.0,
    )
    output_lead_window_ms = clamp_float(
        payload.get("output_lead_window_ms"),
        config.get("default_output_lead_window_ms", 5),
        1.0,
        100.0,
    )
    output_lead_preroll_ms = clamp_float(
        payload.get("output_lead_preroll_ms"),
        config.get("default_output_lead_preroll_ms", 20),
        0.0,
        500.0,
    )
    output_lead_max_gate_ms = clamp_float(
        payload.get("output_lead_max_gate_ms"),
        config.get("default_output_lead_max_gate_ms", 1500),
        10.0,
        5000.0,
    )
    speed = clamp_float(payload.get("speed"), config.get("default_speed", 1.0), 0.25, 4.0)
    cfg_value = clamp_float(payload.get("cfg_value"), config.get("default_cfg_value", 2.0), 0.1, 10.0)
    inference_timesteps = clamp_int(
        payload.get("inference_timesteps"),
        config.get("default_inference_timesteps", 20),
        1,
        100,
    )
    max_generate_length = clamp_int(
        payload.get("max_new_tokens"),
        config.get("default_max_generate_length", 2000),
        16,
        4096,
    )
    preference = str(config.get("engine_preference", "auto")).strip().lower()
    temperature = clamp_float(
        payload.get("temperature"),
        config.get("default_temperature", 1.0),
        0.1,
        2.0,
    )
    streaming = coerce_bool(payload.get("streaming"), bool(config.get("default_streaming", True)))
    response_format = str(payload.get("format") or "wav").strip().lower()
    if response_format not in {"wav", "base64"}:
        response_format = "wav"
    streaming_prefix_len = clamp_int(
        payload.get("streaming_prefix_len"),
        config.get("default_streaming_prefix_len", 2),
        2,
        8,
    )
    seed = None
    raw_seed = payload.get("seed")
    if raw_seed not in (None, ""):
        seed = clamp_int(raw_seed, 0, 0, 2_147_483_647)

    lora_payload_present = any(key in payload for key in ("lora_enabled", "lora_path", "lora_name", "lora"))
    if lora_payload_present:
        raw_lora = payload.get("lora_path") or payload.get("lora_name") or payload.get("lora") or ""
    else:
        raw_lora = config.get("default_lora_path", "")
    lora_enabled = coerce_bool(payload.get("lora_enabled"), bool(str(raw_lora or "").strip()))
    lora_path = resolve_lora_path(raw_lora, config) if lora_enabled else ""
    lora_name = _lora_display_name(Path(lora_path)) if lora_path else ""

    final_text = build_controlled_text(raw_text, control)
    if preference == "nano_vllm":
        nano_vllm_max_model_len = clamp_int(config.get("nano_vllm_max_model_len"), 1536, 512, 32768)
        prompt_token_estimate = max(32, len(final_text) // 3)
        if requested_mode == "ultimate_clone":
            prompt_token_estimate += max(24, len(str(reference.get("text") or "").strip()) // 3)
        prompt_token_estimate += 32
        available_context_cap = max(64, nano_vllm_max_model_len - prompt_token_estimate)
        max_generate_length = min(
            max_generate_length,
            available_context_cap,
            estimate_nano_vllm_max_generate_length(final_text),
        )

    return GenerationRequest(
        text=final_text,
        mode=requested_mode,
        control=control,
        normalize=normalize,
        streaming=streaming,
        speed=speed,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        max_generate_length=max_generate_length,
        temperature=temperature,
        use_memory_cache=use_memory_cache,
        denoise_reference=denoise_reference,
        trim_reference_silence=trim_reference_silence,
        reference_trim_mode=reference_trim_mode,
        reference_lead_threshold_db=reference_lead_threshold_db,
        reference_lead_window_ms=reference_lead_window_ms,
        reference_lead_preroll_ms=reference_lead_preroll_ms,
        trim_output_leading_silence=trim_output_leading_silence,
        output_lead_threshold_db=output_lead_threshold_db,
        output_lead_window_ms=output_lead_window_ms,
        output_lead_preroll_ms=output_lead_preroll_ms,
        output_lead_max_gate_ms=output_lead_max_gate_ms,
        response_format=response_format,
        streaming_prefix_len=streaming_prefix_len,
        lora_path=lora_path,
        lora_name=lora_name,
        reference_id=reference_id,
        reference_audio_bytes=reference_audio_bytes,
        reference_transcript=str(reference.get("text") or "").strip() if isinstance(reference, dict) else "",
        source=source,
        seed=seed,
        fastpath=coerce_bool(payload.get("fastpath"), bool(config.get("fastpath", True))),
        fastpath_prefix=coerce_bool(payload.get("fastpath_prefix"), bool(config.get("fastpath_prefix", True))),
    )


def build_browser_request() -> GenerationRequest:
    form = request.form
    uploaded = request.files.get("reference_audio")
    reference_audio_bytes = uploaded.read() if uploaded and uploaded.filename else None
    payload = {
        "text": form.get("text", ""),
        "mode": form.get("mode", ""),
        "control": form.get("control", ""),
        "normalize": form.get("normalize"),
        "streaming": form.get("streaming"),
        "speed": form.get("speed"),
        "cfg_value": form.get("cfg_value"),
        "inference_timesteps": form.get("inference_timesteps"),
        "max_new_tokens": form.get("max_generate_length"),
        "temperature": form.get("temperature"),
        "streaming_prefix_len": form.get("streaming_prefix_len"),
        "use_memory_cache": form.get("use_memory_cache"),
        "denoise_reference": form.get("denoise_reference"),
        "trim_reference_silence": form.get("trim_reference_silence"),
        "reference_trim_mode": form.get("reference_trim_mode"),
        "reference_lead_threshold_db": form.get("reference_lead_threshold_db"),
        "reference_lead_window_ms": form.get("reference_lead_window_ms"),
        "reference_lead_preroll_ms": form.get("reference_lead_preroll_ms"),
        "trim_output_leading_silence": form.get("trim_output_leading_silence"),
        "output_lead_threshold_db": form.get("output_lead_threshold_db"),
        "output_lead_window_ms": form.get("output_lead_window_ms"),
        "output_lead_preroll_ms": form.get("output_lead_preroll_ms"),
        "output_lead_max_gate_ms": form.get("output_lead_max_gate_ms"),
        "format": "wav",
        "reference_id": form.get("reference_id"),
        "references": [
            {
                "audio": base64.b64encode(reference_audio_bytes).decode("utf-8") if reference_audio_bytes else "",
                "text": form.get("reference_transcript", ""),
            }
        ],
    }
    return parse_request_payload(payload, source="web")


def build_cache_key(engine_kind: str, req: GenerationRequest) -> Optional[str]:
    if req.reference_audio_bytes is None:
        return None
    digest = sha256_bytes(req.reference_audio_bytes)
    transcript_hash = hashlib.sha256(req.reference_transcript.encode("utf-8")).hexdigest()[:16]
    return "|".join(
        [
            engine_kind,
            req.mode,
            f"lora={req.lora_path or 'base'}",
            digest,
            transcript_hash,
            f"reftrim={req.reference_trim_mode}",
            f"refthreshold={req.reference_lead_threshold_db:g}" if req.reference_trim_mode != "off" else "",
            f"refwindow={req.reference_lead_window_ms:g}" if req.reference_trim_mode != "off" else "",
            f"refpreroll={req.reference_lead_preroll_ms:g}" if req.reference_trim_mode != "off" else "",
        ]
    )


def estimate_nano_vllm_max_generate_length(text: str) -> int:
    collapsed = " ".join(str(text or "").split())
    words = len(collapsed.split())
    punctuation_pauses = sum(collapsed.count(mark) for mark in ".!?;:")
    # VoxCPM2 Nano emits audio patches, not text tokens. One generated patch is
    # roughly 160 ms, so text-token style caps can create long vocalization tails
    # when the model misses its stop flag.
    expected_seconds = max(0.9, (words / 2.7) + (punctuation_pauses * 0.18))
    max_seconds = min(30.0, expected_seconds + 0.9)
    estimate = int(math.ceil(max_seconds / 0.16))
    return max(8, min(256, estimate))


def estimate_nano_vllm_max_audio_seconds(text: str) -> float:
    collapsed = " ".join(str(text or "").split())
    words = len(collapsed.split())
    punctuation_pauses = sum(collapsed.count(mark) for mark in ".!?;:")
    expected_seconds = max(0.9, (words / 2.7) + (punctuation_pauses * 0.18))
    return min(30.0, expected_seconds + 1.1)


def prepare_cached_payload(engine: Any, req: GenerationRequest, *, check_preparation=None, commit_preparation=None) -> tuple[Any, Optional[str], bool]:
    reference_audio_bytes = req.reference_audio_bytes
    reference_was_modified = False
    if reference_audio_bytes is not None and engine.kind == "nano_vllm":
        original_reference_audio_bytes = reference_audio_bytes
        if req.reference_trim_mode != "off":
            reference_audio_bytes = trim_reference_audio_bytes(
                reference_audio_bytes,
                req.reference_trim_mode,
                threshold_dbfs=req.reference_lead_threshold_db,
                window_ms=req.reference_lead_window_ms,
                preroll_ms=req.reference_lead_preroll_ms,
            )
        nano_reference_seconds = clamp_float(
            state.get_config().get("nano_vllm_max_reference_seconds"),
            4.0,
            1.0,
            15.0,
        )
        reference_audio_bytes = clip_audio_bytes(reference_audio_bytes, nano_reference_seconds)
        reference_was_modified = reference_audio_bytes != original_reference_audio_bytes

    cache_key = build_cache_key(engine.kind, req)
    reference_digest = sha256_bytes(reference_audio_bytes) if reference_audio_bytes else None
    derived_reference_id = hashlib.sha256((cache_key or reference_digest or "").encode("utf-8")).hexdigest()[:16] if (cache_key or reference_digest) else None
    reference_id = req.reference_id or derived_reference_id

    if engine.kind == "standard":
        if req.denoise_reference or not req.use_memory_cache:
            return None, reference_id, False
        if req.mode not in {"clone", "ultimate_clone"}:
            return None, reference_id, False
        if cache_key or reference_id:
            cached = state.get_cache_entry(cache_key=cache_key, reference_id=reference_id)
            if cached is not None and cached.engine_kind == engine.kind:
                return cached.payload, cached.reference_id, True
        if reference_audio_bytes is None:
            return None, reference_id, False
        if check_preparation is not None:
            check_preparation()
        payload = {
            "prompt_cache": engine.build_prompt_cache(
                reference_audio_bytes=reference_audio_bytes,
                prompt_text=req.reference_transcript,
                mode=req.mode,
                reference_trim_mode=req.reference_trim_mode,
                reference_lead_threshold_db=req.reference_lead_threshold_db,
                reference_lead_window_ms=req.reference_lead_window_ms,
                reference_lead_preroll_ms=req.reference_lead_preroll_ms,
            )
        }
        if hasattr(engine, "prepare_fastpath_prefix"):
            engine.prepare_fastpath_prefix(payload["prompt_cache"])
        resolved_reference_id = reference_id or uuid.uuid4().hex[:16]
        entry = CacheEntry(
            cache_key=cache_key or resolved_reference_id,
            reference_id=resolved_reference_id,
            engine_kind=engine.kind,
            mode=req.mode,
            payload=payload,
            digest=reference_digest or resolved_reference_id,
            transcript=req.reference_transcript,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        if commit_preparation is not None:
            commit_preparation(entry)
        else:
            state.put_cache_entry(entry)
        return payload, resolved_reference_id, False

    if req.use_memory_cache and (cache_key or reference_id):
        cached = state.get_cache_entry(cache_key=cache_key, reference_id=reference_id)
        if cached is not None and cached.engine_kind == engine.kind:
            return cached.payload, cached.reference_id, True

    payload = None
    if req.mode in {"clone", "ultimate_clone"} and reference_audio_bytes is not None:
        if req.reference_trim_mode != "off" and engine.kind != "nano_vllm":
            reference_audio_bytes = trim_reference_audio_bytes(
                reference_audio_bytes,
                req.reference_trim_mode,
                threshold_dbfs=req.reference_lead_threshold_db,
                window_ms=req.reference_lead_window_ms,
                preroll_ms=req.reference_lead_preroll_ms,
            )

        ref_audio_latents = engine.encode_latents(reference_audio_bytes)
        payload = {"ref_audio_latents": ref_audio_latents}
        if req.mode == "ultimate_clone":
            payload["prompt_latents"] = ref_audio_latents
            payload["prompt_text"] = req.reference_transcript

    if payload is None:
        return None, reference_id, False

    if not req.use_memory_cache:
        return payload, reference_id, False

    resolved_reference_id = reference_id or uuid.uuid4().hex[:16]
    entry = CacheEntry(
        cache_key=cache_key or resolved_reference_id,
        reference_id=resolved_reference_id,
        engine_kind=engine.kind,
        mode=req.mode,
        payload=payload,
        digest=reference_digest or resolved_reference_id,
        transcript=req.reference_transcript,
        created_at=now_iso(),
        updated_at=now_iso(),
    )
    state.put_cache_entry(entry)
    return payload, resolved_reference_id, False


def build_execution_plan(req: GenerationRequest) -> ExecutionPlan:
    config = state.get_config()
    force_standard = bool(req.denoise_reference or req.mode in {"clone", "ultimate_clone"})
    engine = state.engine_manager.ensure_loaded(
        config,
        force_standard=force_standard,
        inference_timesteps=req.inference_timesteps,
        lora_path=req.lora_path,
    )
    before_lora = getattr(engine, "lora_path", "")
    engine.ensure_lora(req.lora_path)
    after_lora = getattr(engine, "lora_path", "")
    if before_lora != after_lora:
        state.clear_cache()
        state.log(f"VoxCPM2 LoRA active: {after_lora or 'base model'} (engine={engine.kind})")
    cached_payload, resolved_reference_id, cache_hit = prepare_cached_payload(engine, req)
    if req.mode in {"clone", "ultimate_clone"} and req.reference_audio_bytes is None and not cache_hit:
        raise ValueError("cached reference_id was not found; resend reference audio")

    def _iterator() -> Iterator[np.ndarray]:
        plan = ExecutionPlan(
            request=req,
            engine_kind=engine.kind,
            sample_rate=int(getattr(engine, "sample_rate", 48000)),
            cache_hit=cache_hit,
            reference_id=resolved_reference_id,
            chunk_iterator_factory=lambda: iter(()),
        )
        setattr(plan, "_cached_payload", cached_payload)
        return engine.generate_iterator(plan)

    plan = ExecutionPlan(
        request=req,
        engine_kind=engine.kind,
        sample_rate=int(getattr(engine, "sample_rate", 48000)),
        cache_hit=cache_hit,
        reference_id=resolved_reference_id,
        chunk_iterator_factory=_iterator,
    )
    setattr(plan, "_cached_payload", cached_payload)
    return plan


def _collect_output(
    plan: ExecutionPlan,
    job_id: str,
    *,
    for_streaming_response: bool,
    start_time: Optional[float] = None,
) -> tuple[bytes, dict[str, Any], Optional[Iterator[bytes]]]:
    start_time = time.perf_counter() if start_time is None else float(start_time)
    first_audio_sec: Optional[float] = None
    raw_first_audio_sec: Optional[float] = None
    chunk_count = 0
    audio_chunks: list[np.ndarray] = []
    stream_metrics: dict[str, Any] = {}
    stream_rate = clamp_int(state.get_config().get("stream_sample_rate"), 48000, 16000, 48000)
    if for_streaming_response and abs(plan.request.speed - 1.0) > 1e-6:
        for_streaming_response = False
    trim_output = bool(plan.request.trim_output_leading_silence)
    trim_dropped_samples = 0
    max_audio_samples: Optional[int] = None
    if plan.engine_kind == "nano_vllm" and plan.request.mode == "ultimate_clone":
        max_audio_seconds = estimate_nano_vllm_max_audio_seconds(plan.request.text)
        max_audio_samples = max(1, int(max_audio_seconds * float(plan.sample_rate)))

    def _record_raw_chunks() -> Iterator[np.ndarray]:
        nonlocal raw_first_audio_sec
        with temporary_generation_seed(plan.request.seed):
            for raw_chunk in plan.chunk_iterator_factory():
                if raw_first_audio_sec is None:
                    raw_first_audio_sec = time.perf_counter() - start_time
                yield raw_chunk

    def _iter_stream() -> Iterator[bytes]:
        nonlocal first_audio_sec, chunk_count
        emitted_samples = 0
        try:
            source_chunks = _record_raw_chunks()
            if trim_output:
                source_chunks = iter_lead_trimmed_audio(
                    source_chunks,
                    plan.sample_rate,
                    threshold_dbfs=plan.request.output_lead_threshold_db,
                    window_ms=plan.request.output_lead_window_ms,
                    preroll_ms=plan.request.output_lead_preroll_ms,
                    max_gate_ms=plan.request.output_lead_max_gate_ms,
                )
            for chunk in source_chunks:
                chunk = np.asarray(chunk, dtype=np.float32)
                if max_audio_samples is not None:
                    remaining = max_audio_samples - emitted_samples
                    if remaining <= 0:
                        break
                    if chunk.size > remaining:
                        chunk = chunk[:remaining]
                if first_audio_sec is None:
                    first_audio_sec = time.perf_counter() - start_time
                chunk_count += 1
                state.update_job(
                    job_id,
                    status="streaming",
                    progress=min(92, 42 + chunk_count * 8),
                    message=f"Streaming audio chunk {chunk_count}",
                )
                stream_audio = resample_audio(chunk, plan.sample_rate, stream_rate)
                yield float_to_pcm16(stream_audio)
                emitted_samples += int(chunk.size)
        finally:
            release_cuda_cache_if_large()
            if first_audio_sec is None:
                first_audio_sec = time.perf_counter() - start_time
            total_time_sec = time.perf_counter() - start_time
            stream_metrics.update(
                {
                    "total_time_sec": round(total_time_sec, 3),
                    "first_audio_sec": round(first_audio_sec or total_time_sec, 3),
                    "raw_first_audio_sec": round(raw_first_audio_sec or first_audio_sec or total_time_sec, 3),
                    "chunks": chunk_count,
                }
            )

    if for_streaming_response:
        stream_metrics.update(
            {
            "engine": plan.engine_kind,
            "sample_rate": plan.sample_rate,
            "stream_sample_rate": stream_rate,
            "cache_hit": plan.cache_hit,
            "reference_id": plan.reference_id,
            "lora_path": plan.request.lora_path,
            "lora_name": plan.request.lora_name,
            "reference_trim_mode": plan.request.reference_trim_mode,
            "trim_output_leading_silence": trim_output,
            "output_lead_threshold_db": plan.request.output_lead_threshold_db,
            "output_lead_window_ms": plan.request.output_lead_window_ms,
            "output_lead_preroll_ms": plan.request.output_lead_preroll_ms,
            "output_lead_max_gate_ms": plan.request.output_lead_max_gate_ms,
            "streaming_prefix_len": plan.request.streaming_prefix_len,
            "seed": plan.request.seed,
            }
        )
        return b"", stream_metrics, _iter_stream()

    collected_samples = 0
    for chunk in _record_raw_chunks():
        chunk = np.asarray(chunk, dtype=np.float32)
        if max_audio_samples is not None:
            remaining = max_audio_samples - collected_samples
            if remaining <= 0:
                break
            if chunk.size > remaining:
                chunk = chunk[:remaining]
        if first_audio_sec is None:
            first_audio_sec = time.perf_counter() - start_time
        audio_chunks.append(np.asarray(chunk, dtype=np.float32))
        collected_samples += int(chunk.size)
        chunk_count += 1
        state.update_job(
            job_id,
            status="generating",
            progress=min(94, 40 + chunk_count * 7),
            message=f"Generating chunk {chunk_count}",
        )

    if not audio_chunks:
        raise RuntimeError("model returned no audio chunks")

    audio = np.concatenate(audio_chunks).astype(np.float32)
    if abs(plan.request.speed - 1.0) > 1e-6:
        audio = speed_adjust(audio, plan.request.speed)
    if trim_output:
        audio, trim_dropped_samples = trim_leading_audio(
            audio,
            plan.sample_rate,
            threshold_dbfs=plan.request.output_lead_threshold_db,
            window_ms=plan.request.output_lead_window_ms,
            preroll_ms=plan.request.output_lead_preroll_ms,
        )
    wav_bytes = encode_wav_bytes(audio, plan.sample_rate)
    duration_sec = float(audio.shape[0]) / float(plan.sample_rate)
    total_time_sec = time.perf_counter() - start_time
    metrics = {
        "engine": plan.engine_kind,
        "sample_rate": plan.sample_rate,
        "duration_sec": round(duration_sec, 3),
        "total_time_sec": round(total_time_sec, 3),
        "first_audio_sec": round(first_audio_sec or total_time_sec, 3),
        "rtf": round(total_time_sec / max(duration_sec, 1e-6), 3),
        "chunks": chunk_count,
        "cache_hit": plan.cache_hit,
        "reference_id": plan.reference_id,
        "lora_path": plan.request.lora_path,
        "lora_name": plan.request.lora_name,
        "seed": plan.request.seed,
        "streaming_prefix_len": plan.request.streaming_prefix_len,
        "reference_trim_mode": plan.request.reference_trim_mode,
        "trim_output_leading_silence": trim_output,
        "output_lead_trim_dropped_ms": round((trim_dropped_samples / float(plan.sample_rate)) * 1000.0, 1),
    }
    return wav_bytes, metrics, None


def run_generation(job: JobRecord, req: GenerationRequest, *, streaming_response: bool) -> tuple[ExecutionPlan, bytes, dict[str, Any], Optional[Iterator[bytes]]]:
    start_time = time.perf_counter()
    state.update_job(job.job_id, status="preparing", progress=8, message="Preparing generation request")
    plan = build_execution_plan(req)
    state.update_job(
        job.job_id,
        status="ready",
        progress=28,
        message=f"Engine ready: {plan.engine_kind}",
        engine=plan.engine_kind,
        cache_hit=plan.cache_hit,
        reference_id=plan.reference_id,
    )
    return (plan, *_collect_output(plan, job.job_id, for_streaming_response=streaming_response, start_time=start_time))


def run_generation_on_worker(job, req, *, streaming_response, admission=None):
    return INFERENCE_WORKER.generate(
        lambda: run_generation(job, req, streaming_response=streaming_response),
        lock=state.generation_lock,
        admission=admission,
    )


def finalize_generation(job: JobRecord, wav_bytes: bytes, metrics: dict[str, Any]) -> str:
    output_path = GENERATED_DIR / f"{job.job_id}.wav"
    output_path.write_bytes(wav_bytes)
    state.complete_job(
        job.job_id,
        output_path=str(output_path),
        metrics=metrics,
    )
    return str(output_path)


def dashboard_payload() -> dict[str, Any]:
    return {
        "config": state.get_config(),
        "engine": state.engine_manager.snapshot(),
        "jobs": state.list_jobs(),
        "cache": state.cache_snapshot(),
        "logs": state.get_logs(),
        "supported_languages": SUPPORTED_LANGUAGES,
        "supported_dialects": SUPPORTED_DIALECTS,
        "themes": THEME_NAMES,
        "loras": list_lora_checkpoints(state.get_config()),
    }


@app.get("/")
def index() -> str:
    return jsonify({"ok": True, "service": "voxcpm2-tts", "docs": "/openapi.json"})


@app.get("/health")
def health() -> Response:
    engine = state.engine_manager._engine
    fast = getattr(engine, "fast", None)
    payload = {"status": "ok", **state.engine_manager.snapshot(),
               "admission": INFERENCE_WORKER.snapshot(),
               "fastpath": None if fast is None else {"flash": fast.use_flash, "compile": fast.compile_mode, "graphs": len(fast._dits) + len(fast._vaes) + len(fast._suffixes) + (fast._g_lm is not None),
                                                    "voices": len(fast._prefixes)}}
    return jsonify(payload)


@app.get("/json")
def openapi_alias() -> Response:
    return jsonify(OPENAPI_SCHEMA)


@app.get("/api/dashboard")
def api_dashboard() -> Response:
    return jsonify(dashboard_payload())


@app.get("/api/config")
def api_config_get() -> Response:
    return jsonify(state.get_config())


@app.get("/api/loras")
def api_loras() -> Response:
    return jsonify({"ok": True, "loras": list_lora_checkpoints(state.get_config())})


@app.post("/api/loras/import")
def api_loras_import() -> Response:
    config = state.get_config()
    raw_name = request.form.get("name") or request.args.get("name") or ""
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        uploaded_files = list(request.files.values())
    if not uploaded_files:
        return jsonify({"ok": False, "error": "No LoRA files uploaded"}), 400
    if not raw_name:
        first_name = Path(str(uploaded_files[0].filename or "uploaded_lora")).stem
        raw_name = first_name
    try:
        name = _safe_lora_name(raw_name)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    import_root = _lora_import_root(config)
    target_dir = (import_root / name).resolve()
    import_root.mkdir(parents=True, exist_ok=True)
    temp_dir = import_root / f".{name}.upload-{uuid.uuid4().hex[:8]}"
    saved_names: set[str] = set()
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
        for storage in uploaded_files:
            filename = Path(str(storage.filename or "")).name
            if filename not in LORA_IMPORT_FILENAMES:
                return jsonify({"ok": False, "error": f"Unsupported LoRA file: {filename}"}), 400
            if filename in saved_names:
                continue
            storage.save(str(temp_dir / filename))
            saved_names.add(filename)
        if not any(filename in saved_names for filename in LORA_WEIGHT_FILENAMES):
            return jsonify({"ok": False, "error": "Upload must include lora_weights.*"}), 400
        if "lora_config.json" not in saved_names:
            return jsonify({"ok": False, "error": "Upload must include lora_config.json"}), 400
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in saved_names:
            shutil.move(str(temp_dir / filename), str(target_dir / filename))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    item = {
        "name": _lora_display_name(target_dir),
        "path": str(target_dir),
        "weights": next((str((target_dir / filename).resolve()) for filename in LORA_WEIGHT_FILENAMES if (target_dir / filename).exists()), ""),
    }
    state.log(f"Imported LoRA {name} to {target_dir}")
    return jsonify({"ok": True, "lora": item, "loras": list_lora_checkpoints(config)})


@app.post("/api/config")
def api_config_set() -> Response:
    body = request.get_json(silent=True) or {}
    allowed = {
        key: body[key]
        for key in DEFAULT_CONFIG.keys()
        if key in body
    }
    config = state.save_config(allowed)
    state.log("Config updated from dashboard")
    return jsonify({"ok": True, "config": config})


@app.get("/api/logs")
def api_logs() -> Response:
    return jsonify({"lines": state.get_logs()})


@app.get("/api/jobs")
def api_jobs() -> Response:
    return jsonify({"jobs": state.list_jobs()})


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str) -> Response:
    job = state.get_job(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    return jsonify({"ok": True, "job": job.to_public()})


@app.get("/api/jobs/<job_id>/audio")
def api_job_audio(job_id: str):
    job = state.get_job(job_id)
    if job is None or not job.output_path or not Path(job.output_path).exists():
        return jsonify({"ok": False, "error": "Audio not found"}), 404
    return send_file(job.output_path, mimetype="audio/wav", as_attachment=False, download_name=f"{job_id}.wav")


@app.post("/api/generate")
@generation_admitted
def api_generate(admission) -> Response:
    try:
        req = build_browser_request()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    job = state.create_job("web", req.mode, req.text)

    def _worker() -> None:
        try:
            state.update_job(job.job_id, status="queued", progress=2, message="Waiting for generation slot")
            # This function already runs on the sole model-owning thread.
            with state.generation_lock:
                plan, wav_bytes, metrics, _ = run_generation(job, req, streaming_response=False)
            state.update_job(job.job_id, engine=plan.engine_kind, progress=96, message="Finalizing WAV output")
            finalize_generation(job, wav_bytes, metrics)
            state.log(
                f"Web generation complete: job={job.job_id} mode={req.mode} engine={plan.engine_kind} rtf={metrics.get('rtf')}"
            )
        except Exception as exc:
            state.fail_job(job.job_id, str(exc))
            logger.exception("Web generation failed: job=%s", job.job_id)
            state.log(f"Web generation failed: job={job.job_id} error={exc}")

    def _finished(future) -> None:
        error = "Queued generation cancelled" if future.cancelled() else future.exception()
        if error is not None:
            state.fail_job(job.job_id, str(error))
            state.log(f"Web generation not started: job={job.job_id} error={error}")

    try:
        INFERENCE_WORKER.submit(_worker, admission=admission).add_done_callback(_finished)
    except InferenceBusy as exc:
        state.fail_job(job.job_id, str(exc))
        raise
    return jsonify({"ok": True, "job_id": job.job_id})


@app.post("/v1/tts/prepare")
@generation_admitted(idle_only=True)
def prepare_voice(admission):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "JSON object is required"}), 400
    identity = str(payload.get("preparation_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,96}", identity):
        return jsonify({"ok": False, "error": "A unique preparation_id (8-96 safe characters) is required"}), 400
    ticket = None
    try:
        ticket = PREPARATIONS.begin(identity)
        # Validation reuses the normal request contract; this text is never
        # sent to a generation function or returned as speech.
        payload = dict(payload, text="Reference preparation only.")
        req = parse_request_payload(payload, source="preparation")
        if req.mode not in {"clone", "ultimate_clone"} or req.lora_path or req.denoise_reference or not req.use_memory_cache:
            raise ValueError("Preparation requires a clone mode, memory cache on, denoise off and no LoRA")

        def check() -> None:
            ticket.check()
            if INFERENCE_WORKER.speech_waiting():
                raise InferenceBusy("Speech has priority; prepare the voice when idle")
            state.check_preparation_cache_capacity()

        def commit(entry) -> None:
            def insert():
                check()
                state.put_cache_entry(entry, allow_eviction=False)
            ticket.commit(insert)

        def prepare() -> dict[str, Any]:
            ticket.check()
            if INFERENCE_WORKER.speech_waiting():
                raise InferenceBusy("Speech has priority; prepare the voice when idle")
            with state.generation_lock:
                started = time.perf_counter()
                engine = state.engine_manager.reference_preparation_engine(req)
                cached, reference_id, hit = prepare_cached_payload(
                    engine, req, check_preparation=check, commit_preparation=commit)
                ticket.check()
                if cached is None or not reference_id:
                    raise ValueError("cached reference_id was not found; resend reference audio")
                return {"ok": True, "prepared": True, "preparation_id": identity,
                        "reference_id": reference_id, "cache_hit": hit,
                        "prepare_ms": round((time.perf_counter() - started) * 1000, 3),
                        "engine": engine.kind}

        future = INFERENCE_WORKER.submit(prepare, admission=admission)
        ticket.bind(future)
        result = future.result(timeout=15.0)
        ticket.check()
        response = jsonify(result)
        response.headers["Cache-Control"] = "no-store"
        return response
    except (PreparationCancelled, CancelledError):
        return jsonify({"ok": False, "prepared": False, "preparation_id": identity,
                        "code": "preparation_cancelled", "error": "Voice preparation cancelled"}), 409
    except FutureTimeoutError as exc:
        ticket.cancel()
        raise InferenceBusy("Voice preparation timed out; retry when idle") from exc
    except ValueError as exc:
        return jsonify({"ok": False, "prepared": False, "error": str(exc), "preparation_id": identity}), 400
    finally:
        if ticket is not None:
            PREPARATIONS.finish(ticket)


_WARM_STATE: dict[str, Any] = {}


def _gpu_wake_burst(duration_ms: float = 30.0) -> float:
    """Short GPU load so the 3080 leaves its idle clock state before the sentence ends.

    Between sentences the card drops to P8; the first request after ~15 s idle measured
    ~35 ms slower. A burst at speech onset (seconds before the real request) avoids it.
    """
    import torch

    started = time.perf_counter()
    a = _WARM_STATE.get("a")
    if a is None:
        a = _WARM_STATE["a"] = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    while (time.perf_counter() - started) * 1000.0 < duration_ms:
        for _ in range(8):
            torch.mm(a, a)
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


@app.post("/v1/tts/warm")
@generation_admitted(idle_only=True)
def warm_gpu(admission):
    """Idle-only, no model state touched; busy means the GPU is already awake."""
    try:
        ms = INFERENCE_WORKER.submit(_gpu_wake_burst, admission=admission).result(timeout=2.0)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    response = jsonify({"ok": True, "burst_ms": round(ms, 1)})
    response.headers["Cache-Control"] = "no-store"
    return response


@app.delete("/v1/tts/prepare/<identity>")
def cancel_voice_preparation(identity):
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,96}", identity):
        return jsonify({"ok": False, "error": "Invalid preparation_id"}), 400
    PREPARATIONS.cancel(identity)
    response = jsonify({"ok": True, "cancelled": True, "preparation_id": identity})
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/v1/tts")
@generation_admitted
def compat_tts(admission):
    payload = request.get_json(silent=True) or {}
    try:
        req = parse_request_payload(payload, source="api")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    job = state.create_job("api", req.mode, req.text)
    try:
        state.update_job(job.job_id, status="queued", progress=2, message="Waiting for generation slot")
        plan, wav_bytes, metrics, stream_iter = run_generation_on_worker(
            job, req, streaming_response=req.streaming, admission=admission)
        state.update_job(job.job_id, engine=plan.engine_kind)

        if req.streaming and stream_iter is not None:
            stream_finished = False

            def _close_stream() -> None:
                nonlocal stream_finished
                stream_iter.close()
                if not stream_finished:
                    stream_finished = True
                    state.fail_job(job.job_id, "streaming client disconnected")
                    state.log(f"Streaming client disconnected: job={job.job_id}")

            def _stream() -> Iterator[bytes]:
                nonlocal stream_finished
                try:
                    for chunk in stream_iter:
                        yield chunk
                    stream_metrics = dict(metrics)
                    stream_metrics["streaming"] = True
                    state.complete_job(job.job_id, metrics=stream_metrics)
                    stream_finished = True
                    state.log(
                        f"Streaming generation complete: job={job.job_id} mode={req.mode} engine={plan.engine_kind} temp={req.temperature}"
                    )
                except Exception as exc:
                    stream_finished = True
                    state.fail_job(job.job_id, str(exc))
                    logger.exception("Streaming generation failed: job=%s", job.job_id)
                    state.log(f"Streaming generation failed: job={job.job_id} error={exc}")
                    raise
                except GeneratorExit:
                    stream_finished = True
                    state.fail_job(job.job_id, "streaming client disconnected")
                    logger.warning("Streaming client disconnected: job=%s", job.job_id)
                    state.log(f"Streaming client disconnected: job={job.job_id}")
                    raise
                finally:
                    stream_iter.close()

            response = Response(
                stream_with_context(_stream()),
                mimetype="application/octet-stream",
            )
            if plan.reference_id:
                response.headers["X-Reference-ID"] = plan.reference_id
            if req.lora_name or req.lora_path:
                response.headers["X-LoRA-Name"] = req.lora_name
                response.headers["X-LoRA-Path"] = req.lora_path
            response.headers["X-Job-ID"] = job.job_id
            response.headers["X-Stream-Sample-Rate"] = str(
                clamp_int(state.get_config().get("stream_sample_rate"), 48000, 16000, 48000)
            )
            response.headers["Cache-Control"] = "no-cache, no-transform"
            response.headers["X-Accel-Buffering"] = "no"
            # Normal streaming iteration preserves incremental bytes and also
            # runs close hooks when WSGI closes before the first body chunk.
            response.call_on_close(_close_stream)
            return response

        finalize_generation(job, wav_bytes, metrics)
        state.log(
            f"Compat generation complete: job={job.job_id} mode={req.mode} engine={plan.engine_kind} temp={req.temperature} rtf={metrics.get('rtf')}"
        )
        if req.response_format == "base64":
            return jsonify({
                "audio": base64.b64encode(wav_bytes).decode("utf-8"),
                "reference_id": plan.reference_id,
                "lora_name": req.lora_name,
                "lora_path": req.lora_path,
            })

        response = Response(wav_bytes, mimetype="audio/wav")
        if plan.reference_id:
            response.headers["X-Reference-ID"] = plan.reference_id
        if req.lora_name or req.lora_path:
            response.headers["X-LoRA-Name"] = req.lora_name
            response.headers["X-LoRA-Path"] = req.lora_path
        return response
    except InferenceBusy as exc:
        state.fail_job(job.job_id, str(exc))
        raise
    except Exception as exc:
        state.fail_job(job.job_id, str(exc))
        logger.exception("Compat generation failed: job=%s", job.job_id)
        state.log(f"Compat generation failed: job={job.job_id} error={exc}")
        return jsonify({"error": str(exc)}), 500


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UltraTranslate VoxCPM2 TTS server.")
    parser.add_argument("--host", default=os.getenv("VOXCPM_HOST", "127.0.0.1"))   # this computer only; use 0.0.0.0 for a LAN server
    parser.add_argument("--port", type=int, default=int(os.getenv("VOXCPM_PORT", "8090")))
    parser.add_argument("--model-id", default=os.getenv("VOXCPM_MODEL_ID"))
    parser.add_argument("--engine", choices=["auto", "standard", "nano_vllm"], default=os.getenv("VOXCPM_ENGINE"))
    parser.add_argument("--theme", default=os.getenv("VOXCPM_THEME"))
    parser.add_argument("--default-lora-path", default=os.getenv("VOXCPM_LORA_PATH"))
    parser.add_argument("--stream-sample-rate", type=int, default=int(os.getenv("VOXCPM_STREAM_SAMPLE_RATE")) if os.getenv("VOXCPM_STREAM_SAMPLE_RATE") else None)
    parser.add_argument("--gpu-index", type=int, default=int(os.getenv("VOXCPM_GPU_INDEX")) if os.getenv("VOXCPM_GPU_INDEX") else None)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.getenv("VOXCPM_GPU_MEMORY_UTILIZATION")) if os.getenv("VOXCPM_GPU_MEMORY_UTILIZATION") else None,
    )
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--load-denoiser", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--default-inference-timesteps",
        type=int,
        default=int(os.getenv("VOXCPM_DEFAULT_INFERENCE_TIMESTEPS")) if os.getenv("VOXCPM_DEFAULT_INFERENCE_TIMESTEPS") else None,
    )
    parser.add_argument(
        "--default-cfg-value",
        type=float,
        default=float(os.getenv("VOXCPM_DEFAULT_CFG")) if os.getenv("VOXCPM_DEFAULT_CFG") else None,
    )
    parser.add_argument(
        "--default-max-generate-length",
        type=int,
        default=int(os.getenv("VOXCPM_DEFAULT_MAX_GENERATE_LENGTH")) if os.getenv("VOXCPM_DEFAULT_MAX_GENERATE_LENGTH") else None,
    )
    parser.add_argument(
        "--eager-load",
        action=argparse.BooleanOptionalAction,
        default=coerce_bool(os.getenv("VOXCPM_EAGER_LOAD"), False),
        help="Load and warm the model before accepting requests.",
    )
    parser.add_argument("--log-level", default=os.getenv("VOXCPM_LOG_LEVEL", "info"))
    return parser


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("werkzeug").setLevel(level)


def apply_cli_defaults(args: argparse.Namespace) -> None:
    config_update = {}
    if args.model_id is not None:
        config_update["model_id"] = args.model_id
    if args.theme is not None:
        config_update["theme"] = args.theme
    if args.default_lora_path is not None:
        config_update["default_lora_path"] = args.default_lora_path
    if args.stream_sample_rate is not None:
        config_update["stream_sample_rate"] = args.stream_sample_rate
    if args.gpu_index is not None:
        config_update["gpu_index"] = args.gpu_index
    if args.gpu_memory_utilization is not None:
        config_update["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.optimize is not None:
        config_update["optimize"] = args.optimize
    if args.load_denoiser is not None:
        config_update["load_denoiser"] = args.load_denoiser
    if args.default_inference_timesteps is not None:
        config_update["default_inference_timesteps"] = args.default_inference_timesteps
    if args.default_cfg_value is not None:
        config_update["default_cfg_value"] = args.default_cfg_value
    if args.default_max_generate_length is not None:
        config_update["default_max_generate_length"] = args.default_max_generate_length
    if args.engine:
        config_update["engine_preference"] = args.engine
    if config_update:
        state.save_config(config_update)


def eager_load_model() -> None:
    config = state.get_config()
    inference_timesteps = clamp_int(config.get("default_inference_timesteps"), 20, 4, 20)
    preference = str(config.get("engine_preference", "auto")).strip().lower()
    state.log(f"Preloading VoxCPM engine on startup: preference={preference} timesteps={inference_timesteps}")
    state.engine_manager.ensure_loaded(config, force_standard=False, inference_timesteps=inference_timesteps)

    if os.getenv("VOXCPM_SKIP_SYNTHESIS_WARMUP") == "1":
        state.log("Startup synthesis warmup skipped; model weights are loaded.")
        return

    # Warm the exact streaming path the HTTP API uses so the first live request
    # does not pay the initial graph/setup cost after startup.
    warm_max_generate_length = clamp_int(config.get("default_max_generate_length"), 2000, 200, 4000)
    if preference == "nano_vllm":
        warm_max_generate_length = min(warm_max_generate_length, 256)
    warm_req = GenerationRequest(
        text="Warm up.",
        mode="text",
        control="",
        normalize=False,
        streaming=True,
        speed=1.0,
        cfg_value=clamp_float(config.get("default_cfg_value"), 2.0, 1.0, 5.0),
        inference_timesteps=inference_timesteps,
        max_generate_length=warm_max_generate_length,
        temperature=clamp_float(config.get("default_temperature"), 1.0, 0.1, 2.0),
        use_memory_cache=False,
        denoise_reference=False,
        trim_reference_silence=False,
        reference_trim_mode="off",
        reference_lead_threshold_db=clamp_float(config.get("default_reference_lead_threshold_db"), -40.0, -90.0, -5.0),
        reference_lead_window_ms=clamp_float(config.get("default_reference_lead_window_ms"), 10.0, 1.0, 100.0),
        reference_lead_preroll_ms=clamp_float(config.get("default_reference_lead_preroll_ms"), 20.0, 0.0, 500.0),
        trim_output_leading_silence=False,
        output_lead_threshold_db=clamp_float(config.get("default_output_lead_threshold_db"), -45.0, -90.0, -5.0),
        output_lead_window_ms=clamp_float(config.get("default_output_lead_window_ms"), 5.0, 1.0, 100.0),
        output_lead_preroll_ms=clamp_float(config.get("default_output_lead_preroll_ms"), 20.0, 0.0, 500.0),
        output_lead_max_gate_ms=clamp_float(config.get("default_output_lead_max_gate_ms"), 1500.0, 10.0, 5000.0),
        response_format="wav",
        streaming_prefix_len=clamp_int(config.get("default_streaming_prefix_len"), 2, 2, 8),
        source="startup",
    )
    warm_plan = build_execution_plan(warm_req)
    try:
        for _ in warm_plan.chunk_iterator_factory():
            pass
        state.log(f"Warmup request complete for engine={warm_plan.engine_kind}")
        engine = state.engine_manager._engine
        if hasattr(engine, "warm_fastpath"):
            engine.warm_fastpath(inference_timesteps, clamp_float(config.get("default_cfg_value"), 2.0, 1.0, 5.0))
    except Exception as exc:
        logger.warning("Warmup request failed after engine load; serving will continue: %s", exc)
        state.log(f"Warmup request failed after engine load; continuing without warm cache: {exc}")


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)
    apply_cli_defaults(args)
    if args.eager_load:
        INFERENCE_WORKER.submit(eager_load_model).result()
        state.log("VoxCPM engine preloaded successfully.")
    config = state.get_config()
    state.log(
        f"VoxCPM Flask server ready on {args.host}:{args.port} model={config.get('model_id')} engine_pref={config.get('engine_preference')}"
    )
    app.run(host=args.host, port=args.port, debug=False, threaded=True,
            request_handler=LowLatencyRequestHandler)


if __name__ == "__main__":
    main()
