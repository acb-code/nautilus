from __future__ import annotations
from torch.utils.tensorboard import SummaryWriter

class TBoardLogger:
    def __init__(self, log_dir: str):
        self.w = SummaryWriter(log_dir)

    def scalar(self, key: str, value: float, step: int):
        self.w.add_scalar(key, value, step)

    def flush(self):
        self.w.flush()
