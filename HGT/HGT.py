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
    def __init__(self, in_dim, out_dim, num_node_types, num_edge_types, num_heads, use_norm):
        super(HGTLayer, self).__init__()

        self.in_dim         = in_dim            # Input dimension
        self.out_dim        = out_dim           # Output dimension
        self.num_node_types = num_node_types    # Number of Node types
        self.num_edge_types = num_edge_types    # Number of Edge Types
        self.num_heads      = num_heads         # Number of Attention Heads
        self.head_dim       = out_dim // num_heads
        self.sqrt_head_dim  = math.sqrt(self.head_dim)
        self.use_norm       = use_norm          # (True/False) Use Normalization
        self.attention      = None

        # Creating Learnable Parameters tensors for relation-specific attention weights
        self.rel_priority   = nn.Parameter(torch.ones(self.num_edge_types, self.num_heads))
        self.rel_attention  = nn.Parameter(torch.Tensor(self.num_edge_types, self.num_heads, self.head_dim, self.head_dim))
        self.rel_message    = nn.Parameter(torch.Tensor(self.num_edge_types, self.num_heads, self.head_dim, self.head_dim))
        self.skip           = nn.Parameter(torch.ones(num_node_types))
        self.drop           = nn.Dropout(0.2) # Drop out of 0.2

        # Linear Projections
        self.key_lin_list   = nn.ModuleList()
        self.query_lin_list = nn.ModuleList()
        self.value_lin_list = nn.ModuleList()
        self.agg_lin_list   = nn.ModuleList()

        for i in range(num_node_types):
            self.key_lin_list.append(nn.Linear(in_dim, out_dim))
            self.query_lin_list.append(nn.Linear(in_dim, out_dim))
            self.value_lin_list.append(nn.Linear(in_dim, out_dim))
            self.agg_lin_list.append(nn.Linear(out_dim, out_dim))

    def het_mutal_attention(self, target_node_rep, source_node_rep, key_source_linear, query_source_linear, edge_type_index):
        '''
        Heterogeneous Mutual Attention calculation
        Input:
         - target_node_rep - Node representation of target
         - source_node_rep - Node representation of source
         - key_source_linear   - Linear projection of key source    (nn.ModuleList(), looped nn.Linear layers)
         - query_source_linear - Linear projection of query source  (nn.MOduleList(), looped nn.Linear layers)
         - edge_type_index - index
        Output:
         - res_attention - Tensor storing computed attention coefficients between source and target nodes. 
        '''
        # Apply linear layers for Key (source) and Query (target)
        query_lin_matrix = query_source_linear(target_node_rep).view(-1, self.num_heads, self.head_dim)
        key_lin_matrix = key_source_linear(source_node_rep).view(-1, self.num_heads, self.head_dim)

        # Calculate Relation Attention with Key matrix
        key_lin_attention_matrix = torch.bmm(key_lin_matrix.transpose(1,0), self.rel_attention[edge_type_index]).transpose(1,0)

        # Dot product between new Key matrix and query, then include meta relation triplet tensor divided by root of head dim
        res_attention = (query_lin_matrix * key_lin_attention_matrix).sum(dim = -1) * (self.rel_priority[edge_type_index] / self.sqrt_head_dim)
        return res_attention
    
    def het_message_passing(self, value_source_linear, source_node_rep, edge_type_index):
        '''
        Heterogeneous Message Passing
        Input:
         - value_source_linear - Linear projection of value source  (nn.ModuleList(), looped nn.Linear layers)
         - source_node_rep - Node representation of source
         - edge_type_index - index
        Output:
         - 
        '''
        # Apply Linear Layer
        value_lin_matrix = value_source_linear(source_node_rep).view(-1, self.num_heads, self.head_dim)
        res_message = torch.bmm(value_lin_matrix.transpose(1,0), self.rel_message[edge_type_index]).transpose(1,0)
        return res_message
    
    def target_specific_aggregation(self, aggregated_output, node_input, node_type):
        '''
        Target Specific Aggregation
        x = W[node_type] * gelu(Agg(X)) + x
        Inputs:
        - aggregated_output - output of previous aggregation step
        - node_input - original node input features
        - node_type - the type of the node
        '''
        # GELU activation
        gelu_ag_out = nn.functional.gelu(aggregated_output)
        result = torch.zeros(aggregated_output.size(0), self.out_dim)
        for target_type in range(self.num_node_types):
            index = (node_type == int(target_type))
            if index.sum() == 0:
                continue
            # Applying dropout
            trans_out = self.drop(self.agg_lin_list[target_type](gelu_ag_out[index]))
            # Adding skip connection with learnable weight 
            skip_con = torch.sigmoid(self.skip[target_type])
            result[index] = trans_out *  skip_con + node_input[index] * (1 - trans_out)
        return result
    
    def forward(self, node_input, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_input=node_input, node_type=node_type, edge_type=edge_type)

    def message(self, edge_index_i, source_input_node, source_node_type, target_input_node, target_node_type, edge_type, edge_time):
        '''
        Pytorch Geometric message function under class MessagePassing
        Input:
         - ***_input_node - source/target input node embedding
         - ***_node_type - source/target input node type
         - edge_type - edge type between two nodes 
        '''
        # Create Attention Tensor to store attention coefficients between source and target
        res_attention_tensor = torch.zeros(edge_index_i.size(0), self.num_heads).to(target_input_node.device)
        # Create Message Tensor
        res_message_tensor   = torch.zeros(edge_index_i.size(0), self.num_heads, self.head_dim).to(target_input_node.device)

        for source_type_index in range(self.num_node_types):
            key_source_linear = self.key_lin_list[source_type_index]
            value_source_linear = self.value_lin_list[source_type_index]
            rel_source = (source_node_type == int(source_type_index))

            for target_type_index in range(self.num_node_types):
                query_source_linear = self.query_lin_list[target_type_index]
                rel_target_source = (target_node_type == int(target_type_index)) & rel_source
                for edge_type_index in range(self.num_edge_types):
                    # Meta data relation (edge_type == relation) & (source_type == s) & (target_type == t) 
                    bool_mask_meta = (edge_type == int(edge_type_index)) & rel_target_source
                    
                    # Get relavent Node representations based on rel_edge_target_source
                    source_node_rep = source_input_node[bool_mask_meta]
                    target_node_rep = target_input_node[bool_mask_meta]

                    # Heterogenous Mutual Attention
                    res_attention = self.het_mutal_attention(self, target_node_rep, source_node_rep, key_source_linear, query_source_linear, edge_type_index)
                    res_attention_tensor[bool_mask_meta] = res_attention

                    # Heterogenous Message Passing
                    res_message = self.het_message_passing(self, value_source_linear, source_node_rep, edge_type_index)
                    res_message_tensor[bool_mask_meta] = res_message
            
        # Softmax Output
        self.attention = softmax(res_attention_tensor, edge_index_i)
        result = res_message_tensor * self.attention.view(-1, self.num_heads, 1)
        # Possible to delete tensors from memory
        return result.view(-1, self.out_dim)

    def update(self, aggregated_output, node_input, node_type):
        '''
        Pytorch Geometric Update Function
        '''
        return self.target_specific_aggregation(aggregated_output, node_input, node_type)

class HGTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_node_types, num_edge_types, num_heads, num_layers, dropout):
        super(HGTModel, self).__init__()
        self.input_dim      = input_dim
        self.hidden_dim     = hidden_dim
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.num_heads      = num_heads
        self.num_layers     = num_layers

        # ModuleList and Dropout
        self.adapt_features = nn.ModuleList()
        self.hgt_layers = nn.ModuleList()
        self.drop  = nn.Dropout(dropout)

        # Creating common representation of Features
        for i in range(num_node_types):
            self.adapt_features.append(nn.Linear(input_dim, hidden_dim))
        for j in range(num_layers - 1):
            self.hgt_layers.append(HGTLayer(input_dim, hidden_dim, num_node_types, num_edge_types, num_heads, False))
        # Last layer
        self.hgt_layers.append(HGTLayer(input_dim, hidden_dim, num_node_types, num_edge_types, num_heads, False))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        result = torch.zeros(node_feature.size(0), self.hidden_dim)
        for node_type_index in range(self.num_node_types):
            index = (node_type == int(node_type_index))
            if index.sum() == 0:
                continue
            # tanh activation applied to adapted node features
            result[index] = torch.tanh(self.adapt_features[node_type_index](node_feature[index]))
        
        # Apply dropout
        post_drop = self.drop(result)
        del result # clear
        for layer in self.hgt_layers:
            post_drop = layer(post_drop, node_type, edge_index, edge_type)
        return post_drop

