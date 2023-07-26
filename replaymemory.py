import random

class ReplayMemory():
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        return random.sample(self.buffer[:self.size], batch_size)

    def clear(self):
        self.__init__(self.max_size)