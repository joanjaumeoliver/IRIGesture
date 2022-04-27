from typing import Union, Tuple
from dataset.IRIDatasetTemporal import CustomDynamicGraphTemporalSignal


def temporal_dataset_split(
    data_iterator, train_ratio: float = 0.8
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

    return train_iterator, test_iterator
