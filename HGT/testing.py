import torch
from hgt import *
from model import *

# Create dummy input 
node_feat = torch.rand(20, 64) 
node_type = torch.randint(0, 2, (20,))
edge_index = torch.randint(0, 20, (2, 20))
edge_type = torch.randint(0, 5, (20,))
# Create dummy edge time
num_edges = edge_index.size(1) 
edge_time = torch.randint(0, 100, (num_edges,))



# Create layer
# in_dim, out_dim, num_node_types, num_edge_types, num_heads, use_norm, use_rte
layer = HGTLayer(64, 64, 2, 5, 4, False, False)

# message(self, edge_index_i, target_node_input, source_node_input, target_node_type, source_node_type, edge_type, edge_time):
# Message passing
msg = layer.message(edge_index[0], node_feat, node_feat, node_type, node_type, edge_type, 4)
print("Message passing output shape:", msg.shape)

# # Attention 
att = layer.het_mutual_attention(node_feat, node_feat, layer.key_lin_list[0], 
                                layer.query_lin_list[0], 0)
print("Attention output shape:", att.shape) 

# Message aggregation
agg = layer.het_message_passing(layer.value_lin_list[0], node_feat, 0)
print("Aggregation output shape:", agg.shape)

# Skip connection
out = layer.target_specific_aggregation(node_feat, node_feat, node_type)
print("Skip connection output shape:", out.shape)

# Forward pass
print("FORWARD PASS")
out = layer(node_feat, node_type, edge_index, edge_type, edge_time) 
print("Forward pass output shape:", out.shape)

