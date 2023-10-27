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

class HGTLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, num_node_types, num_edge_types, num_heads, head_dim, use_norm):
        super(HGTLayer, self).__init__()

        self.in_dim         = in_dim            # Input dimension
        self.out_dim        = out_dim           # Output dimension
        self.num_node_types = num_node_types    # Number of Node types
        self.num_edge_types = num_edge_types    # Number of Edge Types
        self.num_heads      = num_heads         # Number of Attention Heads
        self.head_dim       = out_dim // num_heads
        self.sqrt_head_dim  = math.sqrt(self.head_dim)
        self.use_norm       = use_norm          # (True/False) Use Normalization

        # Creating Learnable Parameters tensors for relation-specific attention weights
        self.rel_priority   = nn.Parameter()
        self.rel_attention  = nn.Parameter(torch.Tensor(num_edge_types, num_heads, head_dim, head_dim))
        self.rel_message    = nn.Parameter()

        # Linear Projections
        self.key_projs      = nn.ModuleList()
        self.query_projs    = nn.ModuleList()
        self.value_projs    = nn.ModuleList()

        for i in range(num_node_types):
            self.key_projs.append(nn.Linear(in_dim, out_dim))
            self.query_projs.append(nn.Linear(in_dim, out_dim))
            self.value_projs.append(nn.Linear(in_dim, out_dim))

    def het_mutal_attention(self, target_node_rep, source_node_rep, key_source_linear, query_source_linear, edge_type_index):
        '''
        Heterogeneous Mutual Attention calculation
        Input:
         - target_node_rep - Node representation of target
         - source_node_rep - Node representation of source
         - key_source_linear - Linear projection of key source
         - query_source_linear - Linear projection of query source
         - edge_type_index - edge type 
        Output:
         - 
        '''
        query_lin = query_source_linear(target_node_rep).view()
        key_lin = key_source_linear(source_node_rep).view()
        key_weight = torch.bmm(key_lin.transpose(1,0), self.rel_attention[edge_type_index].transpose(1,0))
        res_attention = (query_lin * key_weight)  * (self.rel_priority[edge_type_index] / self.sqrt_head_dim)
        return res_attention
    
    def het_message_passing(self):

        return
    
    
    def forward(self):
        return self.propagate()

    def message(self, source_input_node, source_node_type, target_input_node, target_node_type, edge_type, edge_time):
        '''
        Pytorch Geometric message function under class MessagePassing
        Input:
         - ***_input_node - 
         - ***_node_type - 
         - edge_type -         
        '''
        # Create Attention Tensor
        res_attention_tensor = torch.zeros()
        # Create Message Tensor


        for source_type_index in range(self.num_node_types):
            key_source_linear = self.key_projs[source_type_index]
            value_source_linear = self.key_projs[source_type_index]
            rel_source = (source_node_type == int(source_type_index))

            for target_type_index in range(self.num_node_types):
                query_source_linear = self.query_projs[target_type_index]
                rel_target_source = (target_node_type == int(target_type_index)) & rel_source
                for edge_type_index in range(self.num_edge_types):
                    # Meta data relation, AND between relavent source, target, and edge types. 
                    rel_edge_target_source = (edge_type == int(edge_type_index)) & rel_target_source
                    
                    # Get relavent Node representations based on rel_edge_target_source
                    source_node_rep = source_input_node[rel_edge_target_source]
                    target_node_rep = target_input_node[rel_edge_target_source]

                    # Heterogenous Mutual Attention
                    res_attention = self.het_mutal_attention()


                    # Heterogenous Message Passing

    def update(self):
        '''
        
        
        '''




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

    def key_linear(self, in_dim, out_dim, num_types):
        '''
        Projection of the source node
        Qi(s) = Q-Linear (H^l-1[s])
        Input:
         - in_dim - dimension of input node features H^l-1[s] --> d
         - out_dim - output dimension, d/h, h is head cout
         - num_types - number of types
        Output:
         - output - Linear projection of key
        '''
        key_linears = nn.ModuleList()
        for i in range(num_types):
            key_linears.append(nn.Linear(in_dim, out_dim))
        return key_linears
    
    def query_linear(self, in_dim, out_dim, num_types):
        '''
        Projection of the TARGET node
        Qi(t) = Q-Linear (H^l-1[t])
        Input:
         - in_dim - dimension of input node features H^l-1[s] --> d
         - out_dim - output dimension d/h, h is head count
         - num_types - number of types
        Output:
         - output - Linear projection of query
        '''
        query_linears = nn.ModuleList()
        for i in range(num_types):
            query_linears.append(nn.Linear(in_dim, out_dim))
        return query_linears
    
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

    def value_linear(self, in_dim, out_dim, num_types, ):
        '''
        Projection of the sources node(s)
        Qi(t) = Q-Linear (H^l-1[t])
        Input:
         - in_dim - dimension of input node features H^l-1[s] --> d
         - out_dim - output dimension d/h, h is head count
         - num_types - number of types
        Output:
         - output - Linear projection of query
        '''
        value_linears = nn.ModuleList()
        for i in range(num_types):
            value_linears.append(nn.Linear(in_dim, out_dim))
        return value_linears

    def messageHead(self, H, W):
        '''
        Calculation of Message Head
        Input:
         - H - Previous layer input
         - W - Weight matrix
        Output:
         - head - Linear Projection 
        '''

        
        return

    def message(self, head):
        '''
        Concat all h message heads to get Message for each node pair
        
        '''


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