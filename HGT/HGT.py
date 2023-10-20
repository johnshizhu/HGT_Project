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

    def key(self, in_dim, out_dim, node_features):
        '''
        Projection of the source node
        Qi(s) = Q-Linear (H^l-1[s])
        Input:
         - in_dim - dimension of input node features H^l-1[s] --> d
         - out_dim - output dimension, d/h, h is head cout
        Output:
         - output - Linear projection of key
        '''
        Qs = nn.Linear(in_dim, out_dim)
        output = Qs(node_features)
        return output
    
    def query(self, in_dim, out_dim, node_features):
        '''
        Projection of the target node
        Qi(t) = Q-Linear (H^l-1[t])
        Input:
         - in_dim - dimension of input node features H^l-1[s] --> d
         - out_dim - output dimension d/h, h is head count
        Output:
         - output - Linear projection of query
        '''
        Qt = nn.Linear(in_dim, out_dim)
        output = Qt(node_features)
        return output
    
    def attention(self, key, W, query):
        '''
        Heterogeneous Mutual Attention 
               
        Input:
         - key - Linear project of key
         - W - distinct edge-based weight matrix for each edge type
         - query - Linear projection of query
        Output:
         - attention - Attention embedding
        '''
        # Transpose of query
        t_query = torch.t(query)

        # Dot product key, weight, query_transpose (left side)
        temp = torch.dot(key, W)
        left = torch.dot(temp, t_query)

        # right hand side meta relation


        # softmax
        soft = nn.Softmax()
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