'''
Access Dataset locally
Download Dataset using Open Academic Graph
'''
import torch

class Local_Access():

    def __init__(self, file_path):
        data = torch.load(file_path)
        data_zero = data[0]
        self.num_nodes_dict = data_zero.num_nodes_dict   # number of nodes of each type
        self.edge_index_dict = data_zero.edge_index_dict # key: type of edge, value: tensor of 2 rows, row 1: first source, row 2: target
        self.x_dict = data_zero.x_dict                   # individual paper
        self.node_year = data_zero.node_year             # year of the paper
        self.edge_reltype = data_zero.edge_reltype
        self.y_dict = data_zero.y_dict
        return
    
    def get_num_nodes(self):
        return self.num_nodes_dict
    
    def get_edge_index_dict(self):
        return self.edge_index_dict
    
    def get_x_dict(self):
        return self.x_dict
    
    def get_node_year(self):
        return self.node_year
    
    def get_edge_reltype(self):
        return self.edge_reltype
    
    def get_y_dict(self):
        return self.y_dict
    
