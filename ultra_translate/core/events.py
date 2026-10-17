from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

_ids = itertools.count(1)


def new_id(prefix: str = "u") -> str:
    return f"{prefix}{next(_ids)}"


@dataclass
class Line:
    """One row in the conversation feed. Partials and the final share `id`, so the UI replaces in place."""
    id: str
    source: str                  # voice | chat
    speaker_id: str              # session speaker key ("S1") or chat display name
    text: str = ""
    language: str = ""           # ISO code detected/assigned
    translation: str = ""
    final: bool = False
    overlap: bool = False
    t_start: float = 0.0         # seconds on the audio clock
    t_end: float = 0.0
    created: float = field(default_factory=time.time)
    timings: dict = field(default_factory=dict)   # ms per stage, for the latency readout
    dropped: str = ""            # reason when a final turned out to be noise/hallucination


@dataclass
class Speaker:
    id: str                      # "S1", "S2", ... stable for the session
    name: str = ""               # user-assigned or from a saved voiceprint
    color: str = "#7aa2ff"
    language: str = ""           # sticky language once confident
    lang_votes: dict = field(default_factory=dict)
    identity: str = ""           # saved voiceprint id when matched
    match_score: float = 0.0
    utterances: int = 0
    seconds: float = 0.0
    last_seen: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        return self.name or f"Speaker {self.id[1:]}"


PALETTE = ["#7aa2ff", "#ff9e64", "#9ece6a", "#f7768e", "#bb9af7", "#e0af68", "#2ac3de", "#ff7ab2",
           "#73daca", "#c0caf5", "#ffb86c", "#a9b665"]
