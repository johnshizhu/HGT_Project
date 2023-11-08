from local_access import Local_Access
import os
import torch
from hgt import *
from torch_geometric.data import HeteroData

path = '../HGT_Data/dataset/ogbn_mag/processed/geometric_data_processed.pt'

dataset = torch.load(path)

data = dataset[0] # pyg graph object

split_idx = dataset.get_idx_split()
train_paper = split_idx['train']['paper'].numpy()
valid_paper = split_idx['valid']['paper'].numpy()
test_paper  = split_idx['test']['paper'].numpy()

