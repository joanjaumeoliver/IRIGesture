import os
import random
import typing
from pathlib import Path

import sys

import torch
import torch.nn.functional
from torch.utils.tensorboard import SummaryWriter

import utils.tools as tools
from dataset.IRIDatasetTemporal import IRIGestureTemporal
from model.AAGCN import Classifier


def train(categories: typing.List[str], tensorboard_name: str):
    step = 0
    loss_list = []
    acc_list = []
    total_guesses = torch.zeros(0).to(DEVICE)
    total_labels = torch.zeros(0).to(DEVICE)
    for encoder_inputs, labels, paths_idx in train_loader:
        total_labels = torch.cat((total_labels, labels))

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
        total_guesses = torch.cat((total_guesses, guessed_list))
        corrects = torch.flatten((guessed_list == labels).float())
        acc = corrects.sum() / len(corrects)
        acc_list.append(acc.item())

        if step % 5 == 0:
            print("Loss = " + str(sum(loss_list) / len(loss_list)))
            print("Acc = " + str(sum(acc_list) / len(acc_list)))
    scheduler.step()
    print("Epoch {} train CrossEntropyLoss: {:.4f} Acc: {:.4f}".format(epoch, sum(loss_list) / len(loss_list),
                                                                       sum(acc_list) / len(acc_list)))

    writer.add_figure("TrainConfusionMatrix", tools.create_confusion_matrix(total_guesses, total_labels, categories,
                                                                            f'Train-Epoch:{epoch}'), epoch)
    writer.add_scalar('Loss/Train', sum(loss_list) / len(loss_list), epoch)
    writer.add_scalar('Accuracy/Train', sum(acc_list) / len(acc_list), epoch)

    for idx, p in enumerate(model.parameters()):
        if p.grad is not None:
            writer.add_scalar(f'TrainGradients/grad_{idx}', p.grad.norm(), epoch)

    writer.add_hparams({'lr': scheduler.get_last_lr()[0]},
                       {'accuracy': sum(acc_list) / len(acc_list),
                        'loss': sum(loss_list) / len(loss_list)})

    torch.save(model.state_dict(), os.path.join(Path().absolute(), 'checkpoints', f'{tensorboard_name}_Checkpoints',
                                                'Epoch_' + str(epoch) + '.pth'))


def test(dataset_videos_paths: typing.List[str], categories: typing.List[str]):
    model.eval()
    batch = 0
    # Store for analysis
    total_loss = []
    total_acc = []
    total_guesses = torch.zeros(0).to(DEVICE)
    total_labels = torch.zeros(0).to(DEVICE)
    for encoder_inputs, labels, paths_idx in test_loader:
        # Get model predictions
        total_labels = torch.cat((total_labels, labels))
        y_hat = model(encoder_inputs)
        # Mean squared error
        loss = loss_fn(y_hat.float(), labels.long())
        total_loss.append(loss.item())

        y_hat_softmax = torch.softmax(y_hat, dim=1)
        guessed_list = torch.argmax(y_hat_softmax, dim=1)
        total_guesses = torch.cat((total_guesses, guessed_list))
        corrects_list = (guessed_list == labels).float()
        corrects = torch.flatten(corrects_list)
        acc = corrects.sum() / len(corrects)
        total_acc.append(acc.item())

        video_idx = random.choice(paths_idx.tolist())
        idx = (paths_idx == video_idx).nonzero(as_tuple=True)[0]
        video_path = dataset_videos_paths[video_idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_label = categories[int(labels[idx])]
        guessed_label = categories[int(guessed_list[idx])]

        writer.add_video(f'{video_label}/{video_name}', tools.read_video(video_path), batch)
        writer.add_text(f'{video_label}/{video_name}',
                        f'Guessed {guessed_label}', batch)

        batch += 1

    print('Test CrossEntropyLoss: {:.4f} Acc: {:.4f}'.format(sum(total_loss) / len(total_loss),
                                                             sum(total_acc) / len(total_acc)))
    writer.add_figure("TestConfusionMatrix", tools.create_confusion_matrix(total_guesses, total_labels, categories,
                                                                           f'Test-Epoch:{epoch}'), epoch)
    writer.add_scalar('Loss/Test', sum(total_loss) / len(total_loss), epoch)
    writer.add_scalar('Accuracy/Test', sum(total_acc) / len(total_acc), epoch)


tools.seed_everything()

DEVICE = torch.device(sys.argv[1])
shuffle = True
batch_size = 32

loader = IRIGestureTemporal(os.path.join(Path().absolute(), 'dataset'), dataTypes="All", token=sys.argv[2])
dataset = loader.get_all_dataset()
dataset.shuffle()
train_dataset, test_dataset = tools.temporal_dataset_split(dataset, train_ratio=0.95)

print("Dataset type:  ", dataset)
print("Number of samples / sequences: ", len(set(dataset)))

print("Number of train buckets: ", len(set(train_dataset)))
print("Number of test buckets: ", len(set(test_dataset)))

# Creating Data loaders
train_loader = tools.create_data_loaders(train_dataset, batch_size, shuffle, DEVICE)
test_loader = tools.create_data_loaders(test_dataset, batch_size, shuffle, DEVICE)

# Create model and optimize
model = Classifier(edge_index=train_dataset.get_static_edge_index().to(DEVICE), out_channels=len(loader.categories),
                   device=DEVICE.type)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
loss_fn = torch.nn.CrossEntropyLoss()

run_name = f'{input("Add TensorBoard RUN Name")}'
tools.clear_path(os.path.join(Path().absolute(), 'checkpoints', f'{run_name}_Checkpoints'))
writer = SummaryWriter(log_dir=os.path.join('tensorboard/runs', run_name))

model.train()
epoch = 0
max_epochs = 3000
while True:
    if epoch < max_epochs:
        train(categories=loader.categories, tensorboard_name=run_name)
        epoch += 1
        if epoch % 25 == 0:
            test(dataset_videos_paths=test_dataset.videos_paths,
                 categories=loader.categories)
            model.train()
    elif input("Do you want to exit?") == 'Yes':
        break
    else:
        max_epochs = int(input("Set new max number of epochs"))

writer.flush()
writer.close()
