from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
import numpy as np
import torch
import math

class ImageSizeBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last, sampler_meta):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.strategy = sampler_meta.strategy
        self.hmin, self.wmin = sampler_meta.min_hw
        self.hmax, self.wmax = sampler_meta.max_hw
        self.divisor = 32
        np.random.seed(0)

    def generate_height_width(self):
        if self.strategy == 'origin':
            return -1, -1
        h = np.random.randint(self.hmin, self.hmax + 1)
        w = np.random.randint(self.wmin, self.wmax + 1)
        h = (h | (self.divisor - 1)) + 1
        w = (w | (self.divisor - 1)) + 1
        return h, w

    def __iter__(self):
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
