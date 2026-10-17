"""Audio sources -> 16 kHz mono float32 frames of 512 samples (32 ms, Silero's native frame).

WASAPI loopback / input via PyAudioWPatch, or a file played in real time (optionally monitored on an
output device so you can hear what is being tested). Resampling is a single stateful soxr stream, so
there are no block-edge artifacts. Downmix is a true channel mean.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import soxr

SR = 16000
REAL_SCENE = os.environ.get("REAL_SCENE", "real_conversation")   # a real recording + its .diar.json
FRAME = 512
_FAST = False   # tests only: feed files as fast as the consumer drains them


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    rate: int
    loopback: bool
    is_output: bool = False


def _pa():
    import pyaudiowpatch as pyaudio
    return pyaudio, pyaudio.PyAudio()


def list_devices() -> dict[str, list[DeviceInfo]]:
    pyaudio, pa = _pa()
    out = {"loopback": [], "input": [], "output": []}
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d.get("hostApi") != wasapi:
                continue
            name = str(d["name"])
            info = DeviceInfo(i, name, int(d["maxInputChannels"] or d["maxOutputChannels"]),
                              int(d["defaultSampleRate"]), bool(d.get("isLoopbackDevice")))
            if info.loopback:
                out["loopback"].append(info)
            elif d["maxInputChannels"] > 0:
                out["input"].append(info)
            elif d["maxOutputChannels"] > 0:
                info.is_output = True
                out["output"].append(info)
    finally:
        pa.terminate()
    return out


def _strip_loopback(name: str) -> str:
    return name.replace(" [Loopback]", "").strip()


def default_loopback_name() -> str:
    pyaudio, pa = _pa()
    try:
        return str(pa.get_default_wasapi_loopback()["name"])
    finally:
        pa.terminate()


class FrameSink:
    """Re-blocks arbitrary-length 16 kHz audio into FRAME-sized chunks and pushes them to a bounded queue."""

    def __init__(self, out: "queue.Queue[tuple[float, np.ndarray]]"):
        self.out = out
        self.buf = np.zeros(0, dtype=np.float32)
        self.samples = 0                     # audio clock in samples
        self.dropped = 0
        self.level = 0.0

    def push(self, audio16: np.ndarray) -> None:
        if audio16.size == 0:
            return
        self.buf = np.concatenate([self.buf, audio16.astype(np.float32, copy=False)])
        n = len(self.buf) // FRAME
        for k in range(n):
            frame = self.buf[k * FRAME:(k + 1) * FRAME]
            t = self.samples / SR
            self.samples += FRAME
            peak = float(np.max(np.abs(frame)))
            self.level = max(peak, self.level * 0.85)
            try:
                self.out.put_nowait((t, frame.copy()))
            except queue.Full:
                self.dropped += 1            # consumer far behind: drop oldest-first semantics via caller
        self.buf = self.buf[n * FRAME:]


class AudioSource(threading.Thread):
    def __init__(self, out_queue: queue.Queue, kind: str, device: str = "", file_path: str = "",
                 monitor: bool = True, monitor_device: str = "", gain: float = 1.0, on_error=None,
                 on_finished=None):
        super().__init__(daemon=True, name=f"audio-{kind}")
        self.kind = kind
        self.device = device
        self.file_path = file_path
        self.monitor = monitor
        self.monitor_device = monitor_device
        self.gain = gain
        self.comp = lambda: 1.0            # extra gain while the game is ducked (GameDucker.comp)
        self.echo = None                   # EchoCanceller: removes our own read-aloud from the capture
        self.sink = FrameSink(out_queue)
        self.stop_event = threading.Event()
        self.on_error = on_error or (lambda msg: None)
        self.on_finished = on_finished or (lambda: None)
        self.status = "starting"
        self.hold = lambda: False          # demo: pause file playback while this returns True
        self.holds: list[tuple[float, float]] = []
        self.rate = 0
        self.channels = 0

    def stop(self) -> None:
        self.stop_event.set()

    @property
    def level(self) -> float:
        return self.sink.level

    def run(self) -> None:
        try:
            if self.kind == "file":
                self._run_file()
            else:
                self._run_device()
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.status = f"error: {exc}"
            self.on_error(f"Audio {self.kind}: {exc}")
        finally:
            self.on_finished()

    # -- devices ---------------------------------------------------------------------------------
    def _find_device(self, pa, pyaudio):
        if self.kind == "loopback":
            if not self.device:
                return pa.get_default_wasapi_loopback()
            want = _strip_loopback(self.device)
            for d in pa.get_loopback_device_info_generator():
                if _strip_loopback(str(d["name"])) == want:
                    return d
            raise RuntimeError(f"loopback device not found: {self.device}")
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        if not self.device:
            return pa.get_device_info_by_index(wasapi["defaultInputDevice"])
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d.get("hostApi") == wasapi["index"] and d["maxInputChannels"] > 0 and d["name"] == self.device:
                return d
        raise RuntimeError(f"input device not found: {self.device}")

    def _run_device(self) -> None:
        pyaudio, pa = _pa()
        stream = None
        try:
            d = self._find_device(pa, pyaudio)
            self.rate = int(d["defaultSampleRate"])
            self.channels = max(1, int(d["maxInputChannels"]))
            rs = soxr.ResampleStream(self.rate, SR, 1, dtype="float32", quality="HQ")
            block = int(self.rate * 0.02)
            stream = pa.open(format=pyaudio.paFloat32, channels=self.channels, rate=self.rate, input=True,
                             input_device_index=int(d["index"]), frames_per_buffer=block)
            self.status = f"capturing {d['name']} @ {self.rate} Hz x{self.channels}"
            while not self.stop_event.is_set():
                avail = stream.get_read_available()
                if avail < block:
                    time.sleep(0.005)
                    continue
                raw = stream.read(avail, exception_on_overflow=False)
                x = np.frombuffer(raw, dtype=np.float32).reshape(-1, self.channels).mean(axis=1)
                g = self.gain * self.comp()
                if g != 1.0:
                    x = x * g
                y = rs.resample_chunk(x)
                if self.echo is not None:
                    y = self.echo.process(y)
                self.sink.push(y)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            pa.terminate()

    # -- file ------------------------------------------------------------------------------------
    def _run_file(self) -> None:
        audio, rate = load_audio_file(self.file_path)
        self.rate, self.channels = rate, 1
        mono16 = soxr.resample(audio, rate, SR, quality="HQ").astype(np.float32) * self.gain
        player = None
        if self.monitor:
            try:
                import sounddevice as sd
                dev = _sd_output_index(self.monitor_device)
                player = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", device=dev,
                                         blocksize=FRAME, latency="low")
                player.start()
            except Exception as exc:
                self.on_error(f"File monitor playback unavailable: {exc}")
                player = None
        self.status = f"playing file {self.file_path}"
        step = FRAME * 2
        t0 = time.perf_counter()
        pos = 0
        while pos < len(mono16) and not self.stop_event.is_set():
            chunk = mono16[pos:pos + step]
            if _FAST:
                while self.sink.out.qsize() > 64 and not self.stop_event.is_set():
                    time.sleep(0.002)
            elif player is not None:
                player.write(chunk.reshape(-1, 1))      # blocks at real-time pace
            else:
                target = t0 + (pos + len(chunk)) / SR
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            self.sink.push(chunk)
            pos += step
            t_hold = time.perf_counter()
            held_at = self.sink.samples / SR
            held = 0
            while self.hold() and not self.stop_event.is_set() and time.perf_counter() - t_hold < 20:
                time.sleep(FRAME / SR)                       # silence keeps the segmenter's clock running
                self.sink.push(np.zeros(FRAME, np.float32))
                held += FRAME
            if held:
                self.holds.append((held_at, held / SR))     # demo pauses, so merges can ignore them
            if time.perf_counter() - t_hold > 0.05:
                t0 = time.perf_counter() - pos / SR
        # flush the segmenter with a little silence so the last utterance closes
        self.sink.push(np.zeros(int(SR * 1.2), dtype=np.float32))
        if player is not None:
            try:
                player.stop()
                player.close()
            except Exception:
                pass
        self.status = "file finished"


def load_audio_file(path: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        return data.mean(axis=1), int(rate)
    except Exception:
        import av  # mp3/m4a/webm via PyAV (bundled with faster-whisper)
        container = av.open(path)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=48000)
        parts = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                parts.append(out.to_ndarray().reshape(-1))
        container.close()
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)).astype(np.float32), 48000


def _sd_output_index(name: str):
    if not name:
        return None
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    return None
