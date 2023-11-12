import torch
import torch.nn as nn
from hgt import HGTLayer

'''
Implementation of Heterogeneous Graph Transformer Model using multiple HGT layers
Inputs:
 - input_dim        - input dimension
 - hidden_dim       - hidden dimension
 - num_node_types   - number of types of nodes
 - num_edge_types   - number of types of edges
 - num_heads        - number of attention heads per layer
 - num_layers       - number of layers
 - dropout          - dropout rate
'''
class HGTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_node_types, num_edge_types, num_heads, num_layers, dropout, prev_norm = False, last_norm = False, use_rte = False):
        super(HGTModel, self).__init__()
        self.input_dim      = input_dim         # Input dimension
        self.hidden_dim     = hidden_dim        # Hiden dimension
        self.num_node_types = num_node_types    # Number of Node types
        self.num_edge_types = num_edge_types    # Number of Edge types
        self.num_heads      = num_heads         # Number of attention heads
        self.num_layers     = num_layers        # Number of layers

        # ModuleList and Dropout
        self.adapt_features = nn.ModuleList()
        self.hgt_layers = nn.ModuleList()
        self.drop  = nn.Dropout(dropout)

        # Loop over number of node types, create linear layers to adapt input features for each node type
        for i in range(num_node_types):
            self.adapt_features.append(nn.Linear(input_dim, hidden_dim))
        # Loop over the number of layers - 1 and add HGT layer for each
        for j in range(num_layers - 1):
            self.hgt_layers.append(HGTLayer(input_dim, hidden_dim, num_node_types, num_edge_types, num_heads, prev_norm, use_rte))
        # Last layer for controlling normalization, etc. for last layer
        self.hgt_layers.append(HGTLayer(input_dim, hidden_dim, num_node_types, num_edge_types, num_heads, last_norm, use_rte))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        # Initialize tensor of 0s for aggregated node features
        result = torch.zeros(node_feature.size(0), self.hidden_dim)

        for node_type_index in range(self.num_node_types):
            # select nodes of the node_type_index type
            index = (node_type == int(node_type_index))
            if index.sum() == 0:
                continue
            # apply linear layer w/ tanh activation to adapt feature for nodes of that type
            result[index] = torch.tanh(self.adapt_features[node_type_index](node_feature[index]))
        
        # Apply dropout
        post_drop = self.drop(result)
        del result # clear
        for layer in self.hgt_layers:
            post_drop = layer(post_drop, node_type, edge_index, edge_type, edge_time)
        return post_drop

class Classifier(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim
        self.linear        = nn.Linear(hidden_dim,  output_dim)
    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)
    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.hidden_dim, self.output_dim)