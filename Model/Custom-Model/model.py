import torch
import torch.nn as nn
import torch.nn.functional as f

class Model(nn.Module):
    def __init__(self, embedding_size, vocab_size, layers, dropout=0.3):
        super(Model, self).__init__()
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.contexts = nn.ModuleList([
            nn.LSTM(embedding_size, embedding_size, batch_first=True) for _ in range(layers)
        ])
        self.currents = nn.ModuleList([
            nn.LSTM(embedding_size, embedding_size, batch_first=True) for _ in range(layers)
        ])
        self.out = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, context):
        x = self.embedding(x)
        context = self.embedding(context)
        
        carrys = []

        for layer in self.contexts:
            context, carry = layer(context)
            context = f.selu(context)
            context = f.dropout(context, p=self.dropout)
            carrys.append(carry)

        for layer, carry in zip(self.currents, carrys):
            x, _ = layer(context, carry)
            x = f.selu(x)
            x = f.dropout(x, p=self.dropout)

        x = self.out(x)
        x = f.softmax(x, dim=-1)
        return x[:, -1, :]

a = torch.randint(24354, size=(1,10))
b = torch.randint(24354, size=(1,30))
model = Model(512, 24354, 8)
print(model(a, b).shape)