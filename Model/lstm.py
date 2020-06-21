import torch
import torch.nn as nn
import torch.nn.functional as f

import pytorch_lightning as pl

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, length):
        super(Dataset, self).__init__()

        self.data = data
        self.length = length
    
    def __getitem__(self, idx): return self.data[idx:idx + self.length], self.data[idx + self.length]

    def __len__(self): return len(self.data) - self.length

