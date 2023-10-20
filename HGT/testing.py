from local_access import Local_Access
import os

path = '../HGT_Data/dataset/ogbn_mag/processed/geometric_data_processed.pt'

local_data = Local_Access(path)

print(local_data.get_data())