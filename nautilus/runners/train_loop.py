from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TrainState:
    step: int = 0
    episode: int = 0
