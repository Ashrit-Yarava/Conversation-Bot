import tensorflow.data as tfds
import numpy as np


class Dataset:

    def __init__(self, data, length):
        self.data = data
        self.length = length
        self.current_idx = 0

    def __len__(self):
        return len(self.data) - self.length

    def __getitem__(self, idx):
        return self.data[idx:idx + self.length], self.data[idx + self.length]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx < len(self):
            return self[self.current_idx]
        self.current_idx = 0
        return self[self.current_idx]

    def __call__(self):
        return self



data = np.random.randint(1000, size=(20000,))
length = 10
dataset = Dataset(data, length)
tf_dataset = tfds.Dataset.from_generator(dataset, (np.ndarray, np.int64))