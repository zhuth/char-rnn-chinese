import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM = 256
HIDDEN_DIM = 512

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self,  x, hs=None):
        batch = x.shape[0]
        if hs is None:
            hs = Variable(
                torch.zeros(self.hidden_dim, batch, self.hidden_dim)).to(device)

        word_embed = self.word_embeddings(x)  # (batch, len, embed)
        word_embed = word_embed.permute(1, 0, 2)  # (len, batch, embed)
        out, h0 = self.lstm(word_embed.view(x.shape[1], batch, -1))  # (len, batch, hidden)
        
        le, mb, hd = out.shape
        out = out.view(le * mb, hd)
        out = self.linear(out)
        out = out.view(le, mb, -1)
        out = out.permute(1, 0, 2).contiguous()  # (batch, len, hidden)
        return out.view(-1, out.shape[2]), h0


def init_model(corpus, checkpoint=''):
        
    loss_function = nn.CrossEntropyLoss()
    model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, corpus.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    start_epoch = 0

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, loss_function, optimizer, start_epoch

