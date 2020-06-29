import argparse
import torch
import torch.nn.functional as f 

from transformers import AutoModelWithLMHead

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, length):
        super(Dataset, self).__init__()

        self.data = data
        self.length = length

    def __len__(self):
        return len(self.data) - self.length

    def __getitem__(self, idx):
        return self.data[idx:idx + self.length], self.data[idx + self.length]

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Simple trainer script.')
parser.add_argument('data_file', type=str, help='Path to the .pt file.')
parser.add_argument('sequence_length', type=int, help='Length of the sequence for the model.')
parser.add_argument('epochs', type=int, help='Number of epochs to train for.')
parser.add_argument('batch_size', type=int, help='Batch size for training.')
parser.add_argument('base_dir', type=str, help='Path to the directory containing the base model.')
parser.add_argument('save_dir', type=str, help='Path to the directory where the model should be saved.')
args = parser.parse_args()

gpt2 = AutoModelWithLMHead.from_pretrained(args.base_dir)

if use_cuda:
    gpt2 = gpt2.cuda()

optimizer = torch.optim.Adam(gpt2.parameters(), lr=1e-4)
data = torch.load(args.data_file)
dataset = Dataset(data, args.sequence_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

for epoch in range(args.epochs):
    print(f"Epoch: {epoch}/{args.epochs}")
    for idx, (sequence, length) in enumerate(dataloader):
        if use_cuda:
            sequence, length = sequence.cuda(), length.cuda()
        
        optimizer.zero_grad()
        logits = gpt2(sequence)[0][:, -1, :]
        loss = f.cross_entropy(logits, length)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(f"\tIndex: {idx}\t\tLoss: {loss.item()}")

gpt2.save_pretrained(args.save_dir)
