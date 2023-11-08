'''
Implementation of relative Temporal Encoding
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RTE(nn.Module):
    '''
    Inputs:
     - hidden_dim - 
     - max_encode -  
    '''
    def __init__(self, hidden_dim, max_encode = 240):
        super(RTE, self).__init__()
        # Create tensor of 1,2,3,..., max_encode-1
        position = torch.arange(0., max_encode, hidden_dim).unsqueeze(1)
        generate_sin = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        embedding = nn.Embedding(max_encode, hidden_dim)
        embedding.weight.data[:, 0::2] = torch.sin(position * generate_sin) / math.sqrt(hidden_dim)
        embedding.weight.data[:, 1::2] = torch.cos(position * generate_sin) / math.sqrt(hidden_dim)
        self.embedding = embedding
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, t):
        return x + self.linear(self.embedding(t))