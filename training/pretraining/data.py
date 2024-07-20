import os
import torch
import numpy as np
import random
import multiprocessing
from sklearn.model_selection import train_test_split

class DistributedDataLoader:
    def __init__(self, data_path, train=True, max_seq_length=512, batch_size=4, num_processes=1):
        self.data_path = data_path
        fs = sorted(os.listdir(data_path))
        if train:
            self.fs = sorted(fs)[:int(len(fs) * 0.9)]
        else:
            self.fs = sorted(fs)[int(len(fs) * 0.9):]
        self.max_seq_length = max_seq_length
        self.current_shard = None
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.offset = self.batch_size * self.max_seq_length
        self.num_proceeses = num_processes
        self.batch_idx = 0
        self.reset()

    def reset(self):
        if self.current_shard != 0:
            self.current_shard = 0
            with open(os.path.join(self.data_path, random.choice(self.fs)), "rb") as f:
                self.tokens = np.frombuffer(f.read(), dtype=np.uint16)
        self.steps = self.tokens.shape[0] // (self.max_seq_length * self.batch_size)
        self.idx = 0
        self.max_steps = self.tokens.shape[0] // (self.max_seq_length * self.batch_size)

    def advance(self):
        filename = os.path.join(self.data_path, random.choice(self.fs))
        with open(filename, "rb") as f:
            self.tokens = np.frombuffer(f.read(), dtype=np.uint16)
        self.max_steps = self.tokens.shape[0] // (self.max_seq_length * self.batch_size)
        self.idx = 0
        # print(filename, self.max_steps)

    def next_batch(self):
        buf = self.tokens[self.idx * self.offset: self.idx * self.offset + (self.offset + 1)]
        buf = torch.tensor(buf.astype(np.uint16), dtype=torch.long)
        x = (buf[:-1]).view(self.batch_size, self.max_seq_length)
        y = buf[1:].view(self.batch_size, self.max_seq_length)
        self.idx += 1
        self.batch_idx += 1
        if self.max_steps * self.offset <= self.idx * self.offset * self.num_processes + 2:
            self.advance()
        return x, y

def next_step(o):
    _ = o.next_batch()

if __name__ == "__main__":
    num_workers = multiprocessing.cpu_count()
    ds = DistributedDataLoader("/scratch/npattab1/llm_data/tokens", True, 10, 4, 1)
    for _ in range(10_000):
        _ = ds.next_batch()

    print(ds.batch_idx)
    print(_[0])
    print(_[1])

