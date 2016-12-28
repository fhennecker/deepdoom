import random


class ReplayMemory():
    def __init__(self, min_size, max_size):
        self.episodes = []
        self.min_size, self.max_size = min_size, max_size

    @property
    def full(self):
        return len(self.episodes) >= self.max_size

    @property
    def initialized(self):
        return len(self.episodes) >= self.min_size

    def add(self, episode):
        if self.full:
            self.episodes.pop(0)
        self.episodes.append(episode)

    def sample(self, batch_size, sequence_length):
        def take_seq():
            episode = random.choice(self.episodes)
            start = random.randint(0, len(episode)-sequence_length)
            return episode[start:start+sequence_length]
        return [take_seq() for b in range(batch_size)]
