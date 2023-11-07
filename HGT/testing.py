from local_access import Local_Access
import os
import torch
from hgt import *
from torch_geometric.data import HeteroData

path = '../HGT_Data/dataset/ogbn_mag/processed/geometric_data_processed.pt'

local_data = Local_Access(path)

print(len(local_data.get_num_nodes()))



