from datetime import datetime
from abc import abstractmethod


class Writer:
    def __init__(self):
        self.mode = None
        self.step = None
        self.timer = None

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    @abstractmethod
    def add_scalar(self, scalar_name, scalar):
        pass
