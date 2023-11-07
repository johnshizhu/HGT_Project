'''
This code opens the Microsoft Academic Graph Dataset and trains HGT

'''
import torch
from hgt import *
from local_access import *
from torch_geometric.loader import DataLoader

print("Microsoft Academic Graph Dataset Experiment")
print("")

path = '../HGT_Data/dataset/ogbn_mag/processed/geometric_data_processed.pt'
local_data = Local_Access(path)

num_node_types = len(local_data.get_num_nodes())
print(f'Number of Nodes types is: {num_node_types}')
num_edge_types = len(local_data.get_edge_index_dict())
print(f'Number of Edge types is: {num_edge_types}')
num_heads = 6
print(f'Number of Attention Heads Per layer is: {num_heads}')
num_layers = 3
print(f'Number of Layers is: {num_layers}')
dropout_rate = 0.2
print(f'Dropout rate is: {dropout_rate}')
paper_features = local_data.get_x_dict()['paper']
print("paper features saved")
input_dim = paper_features.shape[1]
print(f'Input dim is: {input_dim}')
hidden_dim = 256
print(f'Hidden dim is: {hidden_dim}')

model = HGTModel(input_dim, 
                 hidden_dim, 
                 num_node_types, 
                 num_edge_types, 
                 num_heads, 
                 num_layers, 
                 dropout = 0.2)
print("")
print(f'Model is: {model}')
print("")

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
print(f'Optimizer is: {optimizer}')
print("")

loader = DataLoader(local_data.get_data(), batch_size = 32, shuffle = True)
print(f'Dataloader is: {loader}')
print("")










