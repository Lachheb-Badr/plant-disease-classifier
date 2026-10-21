"""Training callbacks."""


class EarlyStopping:
    """Stops training when the monitored value (validation loss) stops improving.

    `step(value)` returns True when `value` is a new best (use it to save a
    checkpoint). `should_stop` becomes True after `patience` epochs in a row
    without an improvement of at least `min_delta`.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, enabled: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.enabled and self.counter >= self.patience:
            self.should_stop = True
        return False
