'''
Faciliates all loading of data using OGB from internet

'''
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader

# split_idx = dataset.get_idx_split()
# train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
# graph = dataset[0] # pyg graph object

class Loader():

    def __init__(self):
        print('Loading dataset Microsoft Academic Graph')
        dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        print('Completed Loading, now splitting Data')
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        data_graph = dataset[0] # pyg graph object
        self.graph = data_graph
        self.train = train_idx 
        self.valid = valid_idx
        self.test = test_idx
        return
