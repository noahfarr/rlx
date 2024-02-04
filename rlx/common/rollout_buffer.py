import mlx.core as mx


class RolloutBuffer:
    def __init__(self):
        self.buffer = {}

    def add(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.buffer:
                self.buffer[key] = []
            self.buffer[key].append(mx.array(value))

    def clear(self):
        self.buffer = {}

    def get(self, key):
        return mx.array(self.buffer.get(key, None))

    def __getitem__(self, key):
        return self.get(key)

    def __str__(self):
        return str(self.buffer)
