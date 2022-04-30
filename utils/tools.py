from typing import List

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchvision.io
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from torch.backends import cudnn


def seed_everything(seed: int = 1997):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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


def sort_list_by_indices(original_list: List, indices: List[int]) -> List[ndarray]:
    return [i for _, i in sorted(zip(indices, original_list))]
