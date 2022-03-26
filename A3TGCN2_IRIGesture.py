from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import torch
import torch.nn.functional as F
from pathlib import Path
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from torch.utils.tensorboard import SummaryWriter

from dataset.IRIDatasetTemporal import IRIGestureTemporal

seed = 1997
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cpu')
shuffle = True
batch_size = 32

loader = IRIGestureTemporal(os.path.join(Path().absolute(), 'dataset'))
dataset = loader.getAllDataset()
train_dataset, test_dataset = loader.getDataset()
# n frames
# 33 nodes
# 4 features
# 8 outputs
print("Dataset type:  ", dataset)
print("Number of samples / sequences: ", len(set(dataset)))

print("Number of train buckets: ", len(set(train_dataset)))
print("Number of test buckets: ", len(set(test_dataset)))

# Creating Dataloaders
train_input = np.array(train_dataset.features)  # (1669, 4, 15, 10)
train_input = np.transpose(train_input, (0, 2, 1, 3))  # (1669, 15, 4, 10)
train_target = np.array(train_dataset.targets)  # (1669, 10, 8)
train_target = np.transpose(train_target, (0, 2, 1))  # (1669, 8, 10)
train_target = np.argmax(train_target, axis=1)
train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (L=1669, N=15, F=4, T=10)
train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (L=1669, T=10)
train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True)

test_input = np.array(test_dataset.features)  # (425, 4, 15, 10)
test_input = np.transpose(test_input, (0, 2, 1, 3))  # (425, 15, 4, 10)
test_target = np.array(test_dataset.targets)  # (425, 10, 8)
test_target = np.transpose(test_target, (0, 2, 1))  # (425, 8, 10)
test_target = np.argmax(test_target, axis=1)
test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B=425, N=15, F=4, T=10)
test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B=425, T=10)
test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# Making the model 
class AttentionGAT(torch.nn.Module):
    def __init__(self, node_features, number_nodes, number_targets, periods, batch_size):
        super(AttentionGAT, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=32, periods=periods,
                            batch_size=batch_size)  # node_features=21, periods=16
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)
        self.softmax = torch.nn.Softmax()
        self.linear2 = torch.nn.Linear(number_nodes, number_targets)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)  # x [b=32, 21, 3, 16]  returns h [b=32, 21, 32]
        h = F.relu(h)
        h = self.linear(h)  # Returns h [b=32, 21, 16]
        h = torch.transpose(h, 2, 1)
        h = self.linear2(h)
        h = torch.transpose(h, 2, 1)
        # h = self.softmax(h) # Implicit in CrossEntropyLoss
        # target (b=32, C=6, F=16)
        return h


# Create model and optimizers
model = AttentionGAT(node_features=4, number_nodes=loader.number_nodes, number_targets=loader.number_targets,
                     periods=loader.number_frames, batch_size=batch_size).to(DEVICE)

# model.load_state_dict(torch.load("C:\_Projects\IRIGesture\A3TGCN2_Checkpoints\Epoch_29.pth"))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Loading the graph once because it's a static graph
static_edge_index = 0
for snapshot in train_dataset:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    break

# Training the model 
model.train()

writer = SummaryWriter()

for epoch in range(300):  # (30, 60):
    step = 0
    loss_list = []
    acc_list = []
    for encoder_inputs, labels in train_loader:
        y_hat = model(encoder_inputs, static_edge_index)  # Get model predictions
        # labels -> [32, 16]
        # y_hat -> [32, 6, 16]
        loss = loss_fn(y_hat.float(), labels.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step = step + 1
        loss_list.append(loss.item())

        corrects = torch.flatten((torch.argmax(y_hat, dim=1) == labels).float())
        acc = corrects.sum() / len(corrects)
        acc_list.append(acc.numpy())

        if step % 10 == 0:
            print("Loss = " + str(sum(loss_list) / len(loss_list)))
            print("Acc = " + str(sum(acc_list) / len(acc_list)))
    print("Epoch {} train CrossEntropyLoss: {:.4f} Acc: {:.4f}".format(epoch, sum(loss_list) / len(loss_list),
                                                                       sum(acc_list) / len(acc_list)))
    writer.add_scalar('Loss/Train', sum(loss_list) / len(loss_list), epoch)
    writer.add_scalar('Accuracy/Train', sum(acc_list) / len(acc_list), epoch)
    torch.save(model.state_dict(), os.path.join(os.path.join(Path().absolute(), 'A3TGCN2_Checkpoints'),
                                                'EpochTemporal_' + str(epoch) + '.pth'))
# Evaluation

model.eval()
step = 0
# Store for analysis
total_loss = []
total_acc = []
for encoder_inputs, labels in test_loader:
    # Get model predictions
    y_hat = model(encoder_inputs, static_edge_index)
    # Mean squared error
    loss = loss_fn(y_hat.float(), labels.long())
    total_loss.append(loss.item())

    corrects = torch.flatten((torch.argmax(y_hat, dim=1) == labels).float())
    acc = corrects.sum() / len(corrects)
    total_acc.append(acc.numpy())

print("Test CrossEntropyLoss: {:.4f} Acc: {:.4f}".format(sum(total_loss) / len(total_loss),
                                                         sum(total_acc) / len(total_acc)))

writer.add_scalar('Loss/Test', sum(total_loss) / len(total_loss), step)
writer.add_scalar('Accuracy/Test', sum(total_acc) / len(total_acc), step)

writer.flush()
writer.close()
