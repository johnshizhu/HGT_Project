import torch
from hgt import *
from model import HGTModel

# Create a simple graph
x = torch.randn(10, 128) # 10 nodes, 128 input features
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]) # connectivity
node_type = torch.zeros(10, dtype=torch.long) 
edge_type = torch.zeros(10, dtype=torch.long)
edge_time = torch.randn(10) 
y = torch.randint(0, 2, (10,)) # binary target


# Model
model = HGTModel(128, 64, 1, 1, 2, 1, 0.5) 
# classifier = Classifier(64, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

print(model)

# Single training iteration
out = model(x, node_type, edge_time, edge_index, edge_type)
# out = classifier(out)
# loss = criterion(out, y)
# loss.backward()
# optimizer.step()

# print("Loss:", loss.item())