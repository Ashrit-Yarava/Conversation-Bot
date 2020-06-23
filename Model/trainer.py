import torch
import torch.nn.functional as f
from transformers import AutoModelWithLMHead


class Dataset:
    def __init__(self, data, length):
        self.data = data
        self.length = length

    def __len__(self): return len(self.data) - self.length

    def __getitem__(self, idx): return self.data[idx:idx + self.length], self.data[idx + self.length]

dataset = torch.load('datasets/TaskMaster.pt')
gpt2 = AutoModelWithLMHead.from_pretrained('distilgpt2')

epochs = 10
learning_rate = 1e-3
batch_size = 64

opt = torch.optim.Adam(gpt2.parameters(), lr=learning_rate)

for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}/{epochs}")
    for idx, (sequence, label) in enumerate(loader):
        opt.zero_grad()
        logits = gpt2(sequence)[0][:, -1, :]
        loss = f.cross_entropy(logits, label)
        loss.backward()
        opt.step()

        if idx % 100 == 0:
            print(f"Index: {idx}\t\tLoss: {loss.item()}")

gpt2.save_pretrained('Model/Checkpoints/')