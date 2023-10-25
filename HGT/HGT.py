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
import math

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
    
    def attentionHead(self, key, W, query, mu, d):
        '''
        Heterogeneous Mutual Attention HEAD calculation
               
        Input:
         - key - Linear project of key
         - W - distinct edge-based weight matrix for each edge type
         - query - Linear projection of query
         - mu - meta relation triplet information of shape (number of node types, number of edge relations, number of node types)
         - d - dimension of the key and query vectors
        Output:
         - attention - Scalar value representing the unnormalized attention score predicted by attention head
        '''
        # Transpose of query
        t_query = torch.t(query)

        # Dot product key, weight, query_transpose (left side)
        temp = torch.dot(key, W)
        left = torch.dot(temp, t_query)

        # right hand side meta relation
        right = mu/math.sqrt(d)

        # element wise multiplication
        return left * right

    
    def attention(self, h, s, e, t):
        '''
        Attention Calculation
        Inputs:
         - h - Number of attention heads
         - s - Source node
         - e - edge type between source and target
         - t - target node
        Outputs:
         - attention - vector of attention score between s and t w/ respect to all neighbors
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
        # POssible object oriented approach to this instead????
        Qv = nn.Linear()

        return
    
    # Target Specific Aggregation

    def aggregate(self, attention, message_passing, res_con):
        '''
        Target Specific Aggregation

        Input:
         - attention - Heterogeneous Mutual Attention
         - message_passing - Heterogeneous Message Passing
         - res_con - Residual connection
        Output:
         - aggregate
        
        '''
        # Element Wise Multiplication

        # Element Wise Addition

        # Activation

        # Linear Layer

        # Element wise Addition w/ Residual Connection

        # aggregate = 

        return