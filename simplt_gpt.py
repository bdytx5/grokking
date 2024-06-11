import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


import math
from argparse import ArgumentParser
from itertools import permutations
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from grokfast import *


class Block(nn.Module):
    """Causal transformer block
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class Decoder(nn.Module):
    """Causal Transformer decoder
    """

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads))

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, x):
        h = self.token_embeddings(x)
        # positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        # h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits

# Dummy text data
text = """
This is a sample text for training the model.
The quick brown fox jumps over the lazy dog.
The model should be able to learn and generate coherent text.
"""

# Tokenize the text
chars = sorted(list(set(text)))
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# Create a dataset
class TextDataset(Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, idx):
        inp = [stoi[char] for char in self.text[idx:idx+self.seq_len]]
        tgt = [stoi[char] for char in self.text[idx+1:idx+self.seq_len+1]]
        return torch.tensor(inp), torch.tensor(tgt)

# Set up the model and optimizer
seq_len = 10
batch_size = 32
model = Decoder(dim=128, num_layers=4, num_heads=4, num_tokens=vocab_size, seq_len=seq_len)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create the data loader
dataset = TextDataset(text, seq_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(10):
    loop = tqdm(data_loader, leave=True)
    for inp, tgt in loop:
        optimizer.zero_grad()
        output = model(inp)
        loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# Save the trained model
torch.save(model.state_dict(), 'model.pth')