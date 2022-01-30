import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2

seed = 1997
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cpu')
shuffle=True
batch_size = 32

from torch_geometric_temporal.dataset import MTMDatasetLoader
loader = MTMDatasetLoader()
dataset = loader.get_dataset(frames=16)
# 16 frames
# 21 nodes
# 3 features
# 6 outputs
print("Dataset type:  ", dataset)
print("Number of samples / sequences: ",  len(set(dataset)))

# Train test split 
from torch_geometric_temporal.signal import temporal_signal_split
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

print("Number of train buckets: ", len(set(train_dataset)))
print("Number of test buckets: ", len(set(test_dataset)))


# Creating Dataloaders
train_input = np.array(train_dataset.features) # (11562, 3, 21, 16)
train_input = np.transpose(train_input, (0,2,1,3)) # (11562, 21, 3, 16)
train_target = np.array(train_dataset.targets) # (11562, 16, 6)
train_target = np.transpose(train_target, (0,2,1)) # (11562, 6, 16)
train_target = np.argmax(train_target, axis=1)
train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (L=11562, N=21, F=3, T=16)
train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (L=11562, C=6, T=16)
train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle,drop_last=True)


test_input = np.array(test_dataset.features) # (2891, 3, 21, 16)
test_input = np.transpose(test_input, (0,2,1,3)) # (11562, 21, 3, 16)
test_target = np.array(test_dataset.targets) # (2891, 16, 6)
test_target = np.transpose(test_target, (0,2,1)) # (11562, 6, 16)
test_target = np.argmax(test_target, axis=1)
test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B=2891, N=21, F=3, T=16)
test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B=2891, C=6, T=16)
test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle,drop_last=True)


# Making the model 
class AttentionGAT(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(AttentionGAT, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, periods=periods,batch_size=batch_size) # node_features=21, periods=16
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)
        self.softmax = torch.nn.Softmax()
        self.linear2 = torch.nn.Linear(21, 6)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # x [b=32, 21, 3, 16]  returns h [b=32, 21, 32]
        h = F.relu(h) 
        h = self.linear(h) #returns h [b=32, 21, 16]
        h = torch.transpose(h, 2, 1)
        h = self.linear2(h)
        h = torch.transpose(h, 2, 1)
        #h = self.softmax(h) #Implicit in CrossEntropyLoss
        #target (b=32, C=6, F=16)
        return h

# Create model and optimizers
model = AttentionGAT(node_features=3, periods=16, batch_size=batch_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Loading the graph once because it's a static graph
for snapshot in train_dataset:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    break;

# Training the model 
model.train()

for epoch in range(30):
    step = 0
    loss_list = []
    acc_list = []
    for encoder_inputs, labels in train_loader:
        y_hat = model(encoder_inputs, static_edge_index)         # Get model predictions
        #labels -> [32, 16]
        #y_hat -> [32, 6, 16]
        loss = loss_fn(y_hat.float(), labels.long()) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step= step+ 1
        loss_list.append(loss.item())
        
        corrects = torch.flatten((torch.argmax(y_hat, dim = 1) == labels).float())
        acc = corrects.sum()/len(corrects)
        acc_list.append(acc.numpy())
        
        if step % 100 == 0 :
            print("Loss = " + str(sum(loss_list)/len(loss_list)))
            print("Acc = " + str(sum(acc_list)/len(acc_list)))
    print("Epoch {} train CrossEntropyLoss: {:.4f} Acc: {:.4f}".format(epoch, sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)))
    torch.save(model.state_dict(), "C:\_Projects\IRIGesture\A3TGCN2_Checkpoints\Epoch_"+str(epoch)+".pth")


## Evaluation

model.eval()
step = 0
# Store for analysis
total_loss = []
total_acc = []
for encoder_inputs, labels in test_loader:
    # Get model predictions
    y_hat = model(encoder_inputs, static_edge_index)
    # Mean squared error
    loss = loss_fn(y_hat, labels)
    total_loss.append(loss.item())
    
    corrects = torch.flatten((torch.argmax(y_hat, dim = 1) == labels).float())
    acc = corrects.sum()/len(corrects)
    acc_list.append(acc.numpy())
    
print("Test CrossEntropyLoss: {:.4f} Acc: {:.4f}".format(sum(total_loss)/len(total_loss), sum(total_acc)/len(total_acc)))