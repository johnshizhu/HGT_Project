'''
Implementation of Heterogeneous Graph Transformer (HGT)
Refer to section 3 of HGT
'''
from ogb_load import Loader
from local_access import Local_Access
import torch
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
        Input - input_graph (Pytorch_Geometric Heterogeneous Graph object)
        '''
        self.graph = input_graph

        return
    
    def layer(self):
        '''

        '''

        return
    
    def attention(self):
        '''
        Heterogeneous Mutual Attention
        Softmax(a(WH^(l-1)[t] || WH^(l-1)[s]))
        ∀s ∈ N(t)
        
        Given target node t, and all its neightbors=[s ∈ N(t)], which might belon got different distributions,
        calculate mutual attention grounded by their meta relations ⟨τ (s),ϕ(e), τ (t)⟩
        
        Input:
         - 
        Output:
         - attention - Attention embedding
        
        '''


        return

    def message(self):
        '''
        Heterogeneous Message Passing
        Message_GAT(s) = WH^(l-1)[s]

        '''

        return
    
    def aggregate(self):
        '''
        Target-Specific Aggregation
        Aggregate (Attention(s,t) DOT (Message (s)))

        '''

        return
    
