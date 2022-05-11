import os
import shutil
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchvision.io
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset

from dataset.IRIDatasetTemporal import CustomDynamicGraphTemporalSignal


def seed_everything(seed: int = 1997):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_video(video_path: str) -> torch.Tensor:
    """
    Read a video with 4D tensor dimensions [time(frame), new_width, new_height, channel]
    and converts it to a 5D tensor [batch-size, time(frame), channel(color), height, width].
    """
    original_video = torchvision.io.read_video(video_path)
    video = np.transpose(original_video[0].numpy()[..., np.newaxis], (4, 0, 3, 1, 2))
    return torch.from_numpy(video)


def create_confusion_matrix(y_pred, y_true, classes, title):
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true.cpu(), y_pred.cpu(), labels=[*range(len(classes))], normalize='true')
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    if classes > 8:
        sn.set(rc={'figure.figsize': (12, 8)})
    s = sn.heatmap(df_cm, annot=True)
    s.set(title=title)
    return s.get_figure()


def sort_list_by_indices(original_list: List, indices: List[int]) -> List[ndarray]:
    return [i for _, i in sorted(zip(indices, original_list))]


def create_data_loaders(dataset: CustomDynamicGraphTemporalSignal, batch_size: int, shuffle: bool,
                        DEVICE: torch.device) -> DataLoader:
    # Prepare features
    features_array = np.array(dataset.features)  # (L, F, N, T)
    features_transposed = np.transpose(features_array, (0, 1, 3, 2))  # (L, F, T, N)
    features_tensor = torch.from_numpy(features_transposed).type(torch.FloatTensor).to(DEVICE)  # (L, F, T, N)

    # Prepare targets
    target_array = np.array(dataset.targets)  # (L, G)
    targets_values = np.argmax(target_array, axis=1)  # (L, )
    target_tensor = torch.from_numpy(targets_values).type(torch.FloatTensor).to(DEVICE)  # (L, )

    # Prepare videos
    videos_list = np.linspace(0, len(dataset.videos_paths), len(dataset.videos_paths), False)  # (L, )
    videos_tensor = torch.from_numpy(videos_list).type(torch.IntTensor).to(DEVICE)  # (L, )

    new_dataset = TensorDataset(features_tensor, target_tensor, videos_tensor)  # (L)
    loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)  # (L/B)

    if len(loader) < 1:
        raise AttributeError('Batch size is too big')
    else:
        return loader


def temporal_dataset_split(
        data_iterator: CustomDynamicGraphTemporalSignal,
        train_ratio: float = 0.8,
) -> Tuple[CustomDynamicGraphTemporalSignal, CustomDynamicGraphTemporalSignal]:
    r"""Function to split a data iterator according to a fixed ratio.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types: * **(train_iterator, test_iterator)** *(tuple of CustomDynamicGraphTemporalSignal Iterators)* -
    Train and test data iterators.
    """
    train_snapshots = int(train_ratio * data_iterator.snapshot_count)

    if type(data_iterator) == CustomDynamicGraphTemporalSignal:
        train_iterator = CustomDynamicGraphTemporalSignal(

            data_iterator.videos_paths[0:train_snapshots],
            data_iterator.edge_indices[0:train_snapshots],
            data_iterator.edge_weights[0:train_snapshots],
            data_iterator.features[0:train_snapshots],
            data_iterator.targets[0:train_snapshots],
        )

        test_iterator = CustomDynamicGraphTemporalSignal(
            data_iterator.videos_paths[train_snapshots:],
            data_iterator.edge_indices[train_snapshots:],
            data_iterator.edge_weights[train_snapshots:],
            data_iterator.features[train_snapshots:],
            data_iterator.targets[train_snapshots:],
        )
    else:
        raise TypeError('Must be CustomDynamicGraphTemporalSignal')

    return train_iterator, test_iterator


def clear_path(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
