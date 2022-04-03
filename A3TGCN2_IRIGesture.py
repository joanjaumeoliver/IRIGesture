import random
import typing

import numpy as np
import os

import torch
import torch.nn.functional
from pathlib import Path

import torchvision.io
from torch.utils.data import TensorDataset
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from dataset.IRIDatasetTemporal import IRIGestureTemporal


def train(tensor_board_enabled: bool):
    step = 0
    loss_list = []
    acc_list = []
    for encoder_inputs, labels, paths_idx in train_loader:
        optimizer.zero_grad()
        y_hat = model(encoder_inputs, static_edge_index, static_weight_index)  # Get model predictions
        loss = loss_fn(y_hat.float(), labels.long())
        loss.backward()
        optimizer.step()
        scheduler.step()
        step = step + 1
        loss_list.append(loss.item())

        corrects = torch.flatten((torch.argmax(y_hat, dim=1) == labels).float())
        acc = corrects.sum() / len(corrects)
        acc_list.append(acc.numpy())

        if step % 25 == 0:
            print("Loss = " + str(sum(loss_list) / len(loss_list)))
            print("Acc = " + str(sum(acc_list) / len(acc_list)))
    print("Epoch {} train CrossEntropyLoss: {:.4f} Acc: {:.4f}".format(epoch, sum(loss_list) / len(loss_list),
                                                                       sum(acc_list) / len(acc_list)))

    if tensor_board_enabled:
        writer.add_scalar('Loss/Train', sum(loss_list) / len(loss_list), epoch)
        writer.add_scalar('Accuracy/Train', sum(acc_list) / len(acc_list), epoch)

        for idx, p in enumerate(model.parameters()):
            writer.add_scalar(f'TrainGradients/grad_{idx}', p.grad.norm(), epoch)

    torch.save(model.state_dict(), os.path.join(Path().absolute(), 'checkpoints', 'A3TGCN2_Checkpoints',
                                                'Epoch_' + str(epoch) + '.pth'))


def test(tensor_board_enabled: bool, dataset_videos_paths: typing.List[str], categories: typing.List[str]):
    model.eval()
    batch = 0
    # Store for analysis
    total_loss = []
    total_acc = []
    for encoder_inputs, labels, paths_idx in test_loader:
        # Get model predictions
        y_hat = model(encoder_inputs, static_edge_index, static_weight_index)
        # Mean squared error
        loss = loss_fn(y_hat.float(), labels.long())
        total_loss.append(loss.item())

        corrects_list = (torch.argmax(y_hat, dim=1) == labels).float()
        corrects = torch.flatten(corrects_list)
        acc = corrects.sum() / len(corrects)
        total_acc.append(acc.numpy())

        if tensor_board_enabled:
            video_idx = random.choice(paths_idx.tolist())
            idx = np.where(paths_idx.numpy() == video_idx)
            video_path = dataset_videos_paths[video_idx]
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_label = categories[int(labels.numpy()[idx, :][0][0][0])]
            video_corrects = torch.flatten(corrects_list[idx, :])
            total_correct_frames = video_corrects.sum()
            if total_correct_frames > 0:
                guessed_label_idx = np.where(video_corrects.numpy() == 1)[0][0]
                guessed_label = categories[int(torch.argmax(y_hat, dim=1).numpy()[idx, :][0][0][guessed_label_idx])]
            else:
                guessed_label = 'Nothing'

            video_result = total_correct_frames / len(video_corrects)
            writer.add_video(f'{video_label}/{video_name}', _read_video(video_path), batch)
            writer.add_text(f'{video_label}/{video_name}',
                            f'Guessed {guessed_label} with an accuracy of: {video_result}', batch)

        batch += 1

    print('Test CrossEntropyLoss: {:.4f} Acc: {:.4f}'.format(sum(total_loss) / len(total_loss),
                                                             sum(total_acc) / len(total_acc)))
    if tensor_board_enabled:
        writer.add_scalar('Loss/Test', sum(total_loss) / len(total_loss), 0)
        writer.add_scalar('Accuracy/Test', sum(total_acc) / len(total_acc), 0)


def _read_video(video_path: str) -> torch.Tensor:
    """
    Read a video with 4D tensor dimensions [time(frame), new_width, new_height, channel]
    and converts it to a 5D tensor [batchsize, time(frame), channel(color), height, width].
    """
    original_video = torchvision.io.read_video(video_path)
    video = np.transpose(original_video[0].numpy()[..., np.newaxis], (4, 0, 3, 1, 2))
    return torch.from_numpy(video)


seed = 1997
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cpu')
shuffle = True
batch_size = 64

loader = IRIGestureTemporal(os.path.join(Path().absolute(), 'dataset'), alsoDownloadVideos=True)
dataset = loader.get_all_dataset()
train_dataset, test_dataset = loader.get_dataset()

print("Dataset type:  ", dataset)
print("Number of samples / sequences: ", len(set(dataset)))

print("Number of train buckets: ", len(set(train_dataset)))
print("Number of test buckets: ", len(set(test_dataset)))

# Creating Dataloaders
train_input = np.array(train_dataset.features)  # (1669, 4, 15, 10)
train_input = np.transpose(train_input, (0, 2, 1, 3))  # (1669, 15, 4, 10)
train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (L=1669, N=15, F=4, T=10)

train_target = np.array(train_dataset.targets)  # (1669, 10, 8)
train_target = np.transpose(train_target, (0, 2, 1))  # (1669, 8, 10)
train_target = np.argmax(train_target, axis=1)
train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (L=1669, T=10)

train_videos = np.linspace(0, len(train_dataset.videos_paths), len(train_dataset.videos_paths), False)
train_videos_tensor = torch.from_numpy(train_videos).type(torch.IntTensor).to(DEVICE)

train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor, train_videos_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True)

test_input = np.array(test_dataset.features)  # (425, 4, 15, 10)
test_input = np.transpose(test_input, (0, 2, 1, 3))  # (425, 15, 4, 10)
test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B=425, N=15, F=4, T=10)

test_target = np.array(test_dataset.targets)  # (425, 10, 8)
test_target = np.transpose(test_target, (0, 2, 1))  # (425, 8, 10)
test_target = np.argmax(test_target, axis=1)
test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B=425, T=10)

test_videos = np.linspace(0, len(test_dataset.videos_paths), len(test_dataset.videos_paths), False)
test_videos_tensor = torch.from_numpy(test_videos).type(torch.IntTensor).to(DEVICE)

test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor, test_videos_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# Making the model 
class AttentionGCN(torch.nn.Module):
    def __init__(self, node_features, number_nodes, number_targets, periods, batch):
        super(AttentionGCN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=32, periods=periods,
                            batch_size=batch)  # node_features=21, periods=16
        # Equals single-shot prediction
        self.F = torch.nn.functional
        self.linear = torch.nn.Linear(32, periods)
        self.softmax = torch.nn.Softmax()
        self.linear2 = torch.nn.Linear(number_nodes, number_targets)

    def forward(self, x, edge_index, edge_weight):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index, edge_weight)  # x [b=32, 21, 3, 16]  returns h [b=32, 21, 32]
        h = self.F.relu(h)
        h = self.linear(h)  # Returns h [b=32, 21, 16]
        h = torch.transpose(h, 2, 1)
        h = self.linear2(h)
        h = torch.transpose(h, 2, 1)
        # h = self.softmax(h) # Implicit in CrossEntropyLoss
        # target (b=32, C=6, F=16)
        return h


# Create model and optimizers
model = AttentionGCN(node_features=4, number_nodes=loader.number_nodes, number_targets=loader.number_targets,
                     periods=loader.number_frames, batch=batch_size).to(DEVICE)
# model = GMAN(L=1, K=8, d=8, num_his=12, bn_decay=0.1, steps_per_day= 288, use_bias=True, mask=False)
# model.load_state_dict(torch.load("C:\_Projects\IRIGesture\A3TGCN2_Checkpoints\Epoch_29.pth"))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
loss_fn = torch.nn.CrossEntropyLoss()

# Loading the graph once because it's a static graph
static_edge_index = 0
static_weight_index = 0
for snapshot in train_dataset:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    static_weight_index = snapshot.edge_attr.to(DEVICE)
    break

writer = SummaryWriter(log_dir=os.path.join('runs', f'{input("Add TensorBoard RUN Name")}'))

model.train()
epoch = 0
max_epochs = 3000
while True:
    if epoch < max_epochs:
        train(tensor_board_enabled=True)
        epoch += 1
        if epoch % 30 == 0:
            test(tensor_board_enabled=True, dataset_videos_paths=test_dataset.videos_paths,
                 categories=loader.categories)
            model.train()
    elif input("Do you want to exit?") == 'Yes':
        break
    else:
        max_epochs = int(input("Set new max number of epochs"))

writer.flush()
writer.close()
