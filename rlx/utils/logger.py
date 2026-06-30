import time

import numpy as np


class Logger:
    def __init__(self, log_interval=100_000):
        self.log_interval = log_interval
        self.start_time = time.time()
        self.next_log = log_interval
        self.returns = []

    def __call__(self, info, step):
        self.returns.extend(info["episode"]["r"][info["_episode"]].tolist())
        if step >= self.next_log:
            self.next_log += self.log_interval
            elapsed = time.time() - self.start_time
            steps_per_second = step / elapsed if elapsed > 0 else 0.0
            mean_return = float(np.mean(self.returns)) if self.returns else float("nan")
            print(
                f"step={step} return={mean_return:.1f} SPS={steps_per_second:,.0f}"
            )
            self.returns = []
