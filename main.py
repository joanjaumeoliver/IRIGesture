import random
import typing

import numpy as np
import os

import pandas as pd
import seaborn as sn
import torch
import torch.nn.functional
from pathlib import Path

import torchvision.io
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from utils.temporal_dataset_split import temporal_dataset_split

from dataset.IRIDatasetTemporal import IRIGestureTemporal
from model.AAGCN import Classifier


def train(tensor_board_enabled: bool, categories: typing.List[str]):
    step = 0
    loss_list = []
    acc_list = []
    total_guesses = np.zeros(0)
    total_labels = np.zeros(0)
    for encoder_inputs, labels, paths_idx in train_loader:
        total_labels = np.concatenate((total_labels, labels.numpy()))

        optimizer.zero_grad()
        y_hat = model(encoder_inputs)  # Get model predictions
        loss = loss_fn(y_hat.float(), labels.long())
        loss.backward()

        optimizer.step()
        step = step + 1
        loss_list.append(loss.item())

        # Softmax is implicit in Loss but not in Acc
        y_hat_softmax = torch.softmax(y_hat, dim=1)
        guessed_list = torch.argmax(y_hat_softmax, dim=1)
        total_guesses = np.concatenate((total_guesses, guessed_list.numpy()))
        corrects = torch.flatten((guessed_list == labels).float())
        acc = corrects.sum() / len(corrects)
        acc_list.append(acc.numpy())

        if step % 5 == 0:
            print("Loss = " + str(sum(loss_list) / len(loss_list)))
            print("Acc = " + str(sum(acc_list) / len(acc_list)))
    scheduler.step()
    print("Epoch {} train CrossEntropyLoss: {:.4f} Acc: {:.4f}".format(epoch, sum(loss_list) / len(loss_list),
                                                                       sum(acc_list) / len(acc_list)))

    if tensor_board_enabled:
        writer.add_figure("TrainConfusionMatrix", __create_confusion_matrix(total_guesses, total_labels, categories,
                                                                            f'Train-Epoch:{epoch}'),
                          epoch)
        writer.add_scalar('Loss/Train', sum(loss_list) / len(loss_list), epoch)
        writer.add_scalar('Accuracy/Train', sum(acc_list) / len(acc_list), epoch)

        for idx, p in enumerate(model.parameters()):
            if p.grad is not None:
                writer.add_scalar(f'TrainGradients/grad_{idx}', p.grad.norm(), epoch)

        writer.add_hparams({'lr': scheduler.get_last_lr()[0]},
                           {'accuracy': sum(acc_list) / len(acc_list),
                            'loss': sum(loss_list) / len(loss_list)})

    torch.save(model.state_dict(), os.path.join(Path().absolute(), 'checkpoints', 'A3TGCN2_Checkpoints',
                                                'Epoch_' + str(epoch) + '.pth'))


def test(tensor_board_enabled: bool, dataset_videos_paths: typing.List[str], categories: typing.List[str]):
    model.eval()
    batch = 0
    # Store for analysis
    total_loss = []
    total_acc = []
    total_guesses = np.zeros(0)
    total_labels = np.zeros(0)
    for encoder_inputs, labels, paths_idx in test_loader:
        # Get model predictions
        total_labels = np.concatenate((total_labels, labels.numpy()))
        y_hat = model(encoder_inputs)
        # Mean squared error
        loss = loss_fn(y_hat.float(), labels.long())
        total_loss.append(loss.item())

        y_hat_softmax = torch.softmax(y_hat, dim=1)
        guessed_list = torch.argmax(y_hat_softmax, dim=1)
        total_guesses = np.concatenate((total_guesses, guessed_list.numpy()))
        corrects_list = (guessed_list == labels).float()
        corrects = torch.flatten(corrects_list)
        acc = corrects.sum() / len(corrects)
        total_acc.append(acc.numpy())

        if tensor_board_enabled:
            video_idx = random.choice(paths_idx.tolist())
            idx = np.where(paths_idx.numpy() == video_idx)
            video_path = dataset_videos_paths[video_idx]
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_label = categories[int(labels.numpy()[idx])]
            guessed_label = categories[int(guessed_list.numpy()[idx])]

            writer.add_video(f'{video_label}/{video_name}', __read_video(video_path), batch)
            writer.add_text(f'{video_label}/{video_name}',
                            f'Guessed {guessed_label}', batch)

        batch += 1

    print('Test CrossEntropyLoss: {:.4f} Acc: {:.4f}'.format(sum(total_loss) / len(total_loss),
                                                             sum(total_acc) / len(total_acc)))
    if tensor_board_enabled:
        writer.add_figure("TestConfusionMatrix", __create_confusion_matrix(total_guesses, total_labels, categories,
                                                                           f'Test-Epoch:{epoch}'),
                          epoch)
        writer.add_scalar('Loss/Test', sum(total_loss) / len(total_loss), epoch)
        writer.add_scalar('Accuracy/Test', sum(total_acc) / len(total_acc), epoch)


def __read_video(video_path: str) -> torch.Tensor:
    """
    Read a video with 4D tensor dimensions [time(frame), new_width, new_height, channel]
    and converts it to a 5D tensor [batch-size, time(frame), channel(color), height, width].
    """
    original_video = torchvision.io.read_video(video_path)
    video = np.transpose(original_video[0].numpy()[..., np.newaxis], (4, 0, 3, 1, 2))
    return torch.from_numpy(video)


def __create_confusion_matrix(y_pred, y_true, classes, title):
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=[*range(len(classes))])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])

    s = sn.heatmap(df_cm, annot=True)
    s.set(title=title)
    return s.get_figure()


seed = 1997
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cpu')
shuffle = True
batch_size = 64

loader = IRIGestureTemporal(os.path.join(Path().absolute(), 'dataset'), alsoDownloadVideos=True,
                            categories=['attention', 'right', 'left', 'stop', 'yes', 'shrug', 'static'])
dataset = loader.get_all_dataset()
# train_dataset, test_dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_dataset_split(dataset, train_ratio=0.95)

print("Dataset type:  ", dataset)
print("Number of samples / sequences: ", len(set(dataset)))

print("Number of train buckets: ", len(set(train_dataset)))
print("Number of test buckets: ", len(set(test_dataset)))

# Creating Dataloaders
train_input = np.array(train_dataset.features)  # (1496, 4, 15, 30)
train_input = np.transpose(train_input, (0, 1, 3, 2))  # (1496, 15, 4, 30)
train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)  # (L=1496, N=15, F=4, T=30)

train_target = np.array(train_dataset.targets)  # (1496, number_of_gestures)
train_target = np.argmax(train_target, axis=1)
train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (L=1669, 1)

train_videos = np.linspace(0, len(train_dataset.videos_paths), len(train_dataset.videos_paths), False)
train_videos_tensor = torch.from_numpy(train_videos).type(torch.IntTensor).to(DEVICE)

train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor, train_videos_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True)

test_input = np.array(test_dataset.features)  # (425, 4, 15, 10)
test_input = np.transpose(test_input, (0, 1, 3, 2))  # (425, 15, 4, 10)
test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)  # (B=425, N=15, F=4, T=10)

test_target = np.array(test_dataset.targets)  # (425, 10, 8)
test_target = np.argmax(test_target, axis=1)
test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B=425, T=10)

test_videos = np.linspace(0, len(test_dataset.videos_paths), len(test_dataset.videos_paths), False)
test_videos_tensor = torch.from_numpy(test_videos).type(torch.IntTensor).to(DEVICE)

test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor, test_videos_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True)

# Loading the graph once because it's a static graph
static_edge_index = 0
static_weight_index = 0
for snapshot in train_dataset:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    static_weight_index = snapshot.edge_attr.to(DEVICE)
    break

# Create model and optimize
model = Classifier(edge_index=static_edge_index)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
loss_fn = torch.nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir=os.path.join('tensorboard/runs', f'{input("Add TensorBoard RUN Name")}'))

model.train()
epoch = 0
max_epochs = 3000
while True:
    if epoch < max_epochs:
        train(tensor_board_enabled=True,
              categories=loader.categories)
        epoch += 1
        if epoch % 25 == 0:
            test(tensor_board_enabled=True,
                 dataset_videos_paths=test_dataset.videos_paths,
                 categories=loader.categories)
            model.train()
    elif input("Do you want to exit?") == 'Yes':
        break
    else:
        max_epochs = int(input("Set new max number of epochs"))

writer.flush()
writer.close()
