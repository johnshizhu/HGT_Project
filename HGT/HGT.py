'''
Implementation of Heterogeneous Graph Transformer (HGT)
Refer to section 3 of HGT
'''
from ogb_load import Loader
from local_access import Local_Access
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.data import HeteroData


class HGT():
    '''
    Attention --> estimates the importance of each source node
    Message --> extracts the message by using only the source node s
    Aggregate --> aggregates neighborhood message by attention weight
    '''
    def __init__(self, input_graph):
        '''
        Input
         - input_graph (Pytorch_Geometric Heterogeneous Graph object)
        '''
        self.graph = input_graph
        return
    
    def layer(self):
        '''

        '''

        return

    # Heterogeneous Mutual Attention

    def key(self, H, s):
        '''
        Projection of the source node
        Qi(s) = Q-Linear (H^l-1[s])
        Input:
         - H - Output of previous layer
         - s - source node
        Output:
         - Linear projection
        '''
        Qs = nn.Linear()

        return
    
    def query(self):
        '''
        Projection of the target node
        Qi(t) = Q-Linear (H^l-1[t])
        '''
        Qt = nn.Linear()

        return
    
    def attention(self, key, W, query):
        '''
        Heterogeneous Mutual Attention 
               
        Input:
         - key - 
         - W - 
         - query - 
        Output:
         - attention - Attention embedding
        '''

        return
    
    # Heterogeneous Message Passing - pass info from source nodes to target nodes

    def message(self, head):
        '''
        Concat all h message heads to get Message for each node pair
        '''


        return
    
    def messageHead(self):

        return
    
    def value(self):
        '''
        Linear Projection of source node s into ith message vector
        '''
        Qv = nn.Linear()

        return
    
    # Target Specific Aggregation

    def aggregate(self):

        return