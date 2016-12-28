import random
import numpy as np


class ReplayMemory():
    def __init__(self, min_size, max_size):
        self.episodes = []
        self.min_size, self.max_size = min_size, max_size

    def __len__(self):
        return sum(map(len, self.episodes))

    @property
    def full(self):
        return len(self) >= self.max_size

    @property
    def initialized(self):
        return len(self) >= self.min_size

    def add(self, episode):
        if self.full:
            self.episodes.pop(0)
        self.episodes.append(episode)

    def sample(self, batch_size, sequence_length):
        def take_seq():
            episode = random.choice(self.episodes)
            start = random.randint(0, len(episode)-sequence_length)
            return [np.array(x) for x in zip(*episode[start:start+sequence_length])]
        return [take_seq() for b in range(batch_size)]
