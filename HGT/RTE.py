'''
Implementation of relative Temporal Encoding
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RTE(nn.Module):
    '''
    Implementation of Relative Temporal Encoding based on original paper implemenation
    '''
    def __init__(self, hidden_dim, max_encode = 240, dropout = 0.2):
        super(RTE, self).__init__()
        # Positions tensor represents positions of each element in sequence
        position = torch.arange(0., max_encode).unsqueeze(1)
        # Values use to compute sinusoid function
        generate_sin = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        
        embedding = nn.Embedding(max_encode, hidden_dim)
        embedding.weight.data[:, 0::2] = torch.sin(position * generate_sin) / math.sqrt(hidden_dim)
        embedding.weight.data[:, 1::2] = torch.cos(position * generate_sin) / math.sqrt(hidden_dim)
        embedding.requires_grad = False
        
        self.embedding = embedding
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, t):
        # Inputs x and the position t
        return x + self.linear(self.embedding(t))