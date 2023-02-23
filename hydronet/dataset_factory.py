# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os.path as osp
from itertools import chain
from typing import Optional, Tuple

import numpy as np
import poptorch
import torch
from torch_geometric.datasets import QM9, HydroNet
from tqdm import tqdm

from .data_utils import dataroot
from .dataloader import create_dataloader
from .packing import (
    PackedDataset,
    hydronet_medium_packing_strategy,
    hydronet_packing_strategy,
    hydronet_small_packing_strategy,
    qm9_packing_strategy,
)
from .transforms import create_transform


def calculate_splits(splits: Tuple[float], num_examples: int):
    """
    Converts the tuple describing dataset splits as a fraction of the full
    dataset into the number of examples to include in each split.

    :param splits (tuple of float): A tuple describing the ratios to
        divide the full dataset into training, validation, and test splits.
    :param num_examples (int): The total number of examples in the full dataset
    """
    assert (
        isinstance(splits, (list, tuple))
        and all(isinstance(s, float) for s in splits)
        and len(splits) == 3
        and sum(splits) == 1.0
    ), (
        f"Invalid splits {splits}. Must be a tuple or list containing "
        "exactly three floats that add up to 1.0."
    )

    splits = torch.tensor(splits)
    splits = torch.round(splits * num_examples).long()
    splits = torch.cumsum(splits, 0)
    return tuple(splits.tolist())


class DatasetFactory:
    """
    DatasetFactory

    Abstract interface for managing the reproducible application of dataset
    transforms and randomized splits. Sub-classes must implement the following:

        * create_dataset: materialize the full dataset
        * strategy: the pre-computed dataset packing strategy

    Typical usage:

        factory = ConcreteFactory(...)
        loader = factory.dataloader(split="train", num_workers=4)
        ...
    """

    def __init__(
        self,
        root: str = "data",
        splits: Optional[Tuple[float]] = None,
        batch_size: int = 1,
        use_packing: bool = False,
        k: int = 8,
        options: Optional[poptorch.Options] = None,
    ):
        """
        :param root (str): The root folder name for storing the dataset.
        :param splits (tuple of float, optional): A tuple describing the ratios
            to divide the full dataset into training, validation, and test
            datasets (default: (0.8, 0.1, 0.1)).
        :param batch_size (int): The data loader batchsize (default: 1)
        :param use_packing (bool): Use packing to minimise padding values
            (default: False).
        :param k (int): Number of neighbors to use for building the k-nearest
            neighbors graph when using packing (default: 8).
        :param options (poptorch.Options, optional): Instance of
            poptorch.Options
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.model_batch_size = self.batch_size
        self.splits = (0.8, 0.1, 0.1) if splits is None else splits
        self.use_packing = use_packing
        self.k = k

        self.options = options
        self.reset()

        if not self.use_packing:
            return

        # The model batch size is determined by the largest pack
        max_num_graphs = self.strategy.max_num_graphs
        self.model_batch_size = max_num_graphs * self.batch_size

    def reset(self):
        """
        resets the state of the factory.

        This method is necessary to ensure that the factory does not maintain a
        persistent reference to a fully materialized dataset.
        """
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def create_dataset(self):
        """
        Abstract method for creating the full dataset used by the factory.
        """
        raise NotImplementedError

    @property
    def strategy(self):
        """
        Abstract property for storing the packing strategy used by the factory.
        """
        raise NotImplementedError

    @property
    def num_nodes(self):
        """
        Abstract property for storing the mapping from index to number of nodes
        """
        raise NotImplementedError

    @property
    def num_nodes_file(self):
        return osp.join(self.root, "processed", "num_nodes.pt")

    def setup_splits(self):
        """
        Setup dataset splits by randomly assigning examples to the training,
        validation, or test splits.
        """
        dataset = self.create_dataset()

        if self.use_packing:
            dataset = PackedDataset(
                dataset, self.strategy, self.num_nodes, shuffle=False
            )

        splits = calculate_splits(self.splits, len(dataset))
        self._train_dataset = dataset[0 : splits[0]]
        self._val_dataset = dataset[splits[0] : splits[1]]
        self._test_dataset = dataset[splits[1] :]

    @property
    def combined_batch_size(self):
        return (
            self.batch_size
            * self.options.device_iterations
            * self.options.replication_factor
            * self.options.Training.gradient_accumulation
        )

    @property
    def train_dataset(self):
        """
        Training split of the dataset
        """
        if self._train_dataset is None:
            self.setup_splits()
        return self._train_dataset

    @property
    def val_dataset(self):
        """
        Validation split of the dataset
        """
        if self._val_dataset is None:
            self.setup_splits()
        return self._val_dataset

    @property
    def test_dataset(self):
        """
        Test split of the dataset
        """
        if self._test_dataset is None:
            self.setup_splits()
        return self._test_dataset

    def dataloader(
        self, split: str = "train", max_workers: int = 4, use_async_loader: bool = True
    ):
        """
        Create a dataloader for the desired split of the dataset.

        :param split (str): The desired dataset split to load.
            (default: "train")
        :param max_workers (int): the maximum number of asynchronous workers
            used by the dataloader. (default: 4).
        :param use_async_loader: Enables poptorch.DataLoaderMode.Async
            (default: True).
        """
        dataset = LazyDataset(self, split)
        num_batches = len(dataset) // self.combined_batch_size
        num_workers = min(num_batches, max_workers)
        buffer_size = num_batches // num_workers if num_workers > 0 else None
        shuffle = True if split == "train" else False
        return create_dataloader(
            dataset,
            self.options,
            use_packing=self.use_packing,
            batch_size=self.batch_size,
            shuffle=shuffle,
            k=self.k,
            num_workers=num_workers,
            buffer_size=buffer_size,
            use_async_loader=use_async_loader,
        )


class QM9DatasetFactory(DatasetFactory):
    def __init__(
        self,
        root: str = "data",
        splits: Optional[Tuple[float]] = None,
        batch_size: int = 1,
        use_packing: bool = False,
        k: int = 8,
        options: Optional[poptorch.Options] = None,
    ):
        self.transform = create_transform(
            use_qm9_energy=True,
            use_standardized_energy=True,
            use_padding=not use_packing,
            max_num_nodes=32,
        )
        super().__init__(root, splits, batch_size, use_packing, k, options)

    def create_dataset(self):
        """
        Create the QM9 dataset. Downloads to the root location and applies the
        dataset transform to a pre-processed version saved to disk.
        """
        return QM9(root=self.root, pre_transform=self.transform)

    @property
    def strategy(self):
        """
        The pre-computed packing strategy for the QM9 dataset.
        """
        return qm9_packing_strategy()

    @property
    def num_nodes(self):
        if osp.exists(self.num_nodes_file):
            return torch.load(self.num_nodes_file)

        dataset = self.create_dataset()
        bar = tqdm(dataset, desc="Calculating number of nodes")
        num_nodes = [g.num_nodes for g in bar]
        num_nodes = torch.tensor(num_nodes)
        torch.save(num_nodes, self.num_nodes_file)
        return num_nodes

    @staticmethod
    def create(batch_size=1, use_packing=False, options=None):
        """
        Static factory method for creating this factory with the root location
        customised for packing/padding. This can help save time downloading and
        processing the dataset for repeated runs.
        """
        suffix = "packed" if use_packing else "padded"

        return QM9DatasetFactory(
            root=dataroot(f"qm9_{suffix}"),
            batch_size=batch_size,
            k=28,
            use_packing=use_packing,
            options=options,
        )


class HydroNetDatasetFactory(DatasetFactory):
    def __init__(
        self,
        root: str = "data",
        splits: Optional[Tuple[float]] = None,
        batch_size: int = 1,
        use_packing: bool = False,
        k: int = 28,
        cutoff: float = 6.0,
        options: Optional[poptorch.Options] = None,
        name: str = "large",
    ):
        self.transform = None

        if not use_packing:
            self.transform = create_transform(
                use_qm9_energy=False,
                use_standardized_energy=False,
                use_padding=True,
                max_num_nodes=96,
            )
        self.name = name
        self.cutoff = cutoff
        super().__init__(root, splits, batch_size, use_packing, k, options)

    @property
    def split_file(self):
        return osp.join(self.root, "processed", f"split_00_{self.name}.npz")

    def create_dataset(self):
        if self.name == "large":
            return HydroNet(root=self.root, pre_transform=self.transform)

        if self.name == "medium":
            return HydroNet(
                root=self.root, clusters=range(3, 26), pre_transform=self.transform
            )

        # small 500k dataset is a random sample of the medium 2.7m one
        dataset = HydroNet(
            root=self.root, clusters=range(3, 26), pre_transform=self.transform
        )

        with np.load(self.split_file) as split_file:
            train_idx = split_file["train_idx"]
            val_idx = split_file["val_idx"]
            all_idx = np.concatenate([train_idx, val_idx])
            dataset = dataset.index_select(all_idx)

        return dataset

    @property
    def strategy(self):
        if self.name == "large":
            return hydronet_packing_strategy()

        if self.name == "medium":
            return hydronet_medium_packing_strategy()

        return hydronet_small_packing_strategy()

    @property
    def num_nodes_file(self):
        return osp.join(self.root, "processed", f"num_nodes_{self.name}.pt")

    @property
    def num_nodes(self):
        if osp.exists(self.num_nodes_file):
            return torch.load(self.num_nodes_file)

        # Cache the num_nodes tensor for packing
        dataset = self.create_dataset()

        if self.name == "small":
            num_nodes_list = [d.num_nodes for d in tqdm(dataset)]
            num_nodes = torch.tensor(num_nodes_list)
        else:
            num_nodes_list = []
            for partition in dataset._partitions:
                num_nodes = 3 * partition.num_clusters
                num_nodes_list.append([num_nodes] * len(partition))

            num_nodes = torch.tensor(list(chain.from_iterable(num_nodes_list)))

        torch.save(num_nodes, self.num_nodes_file)
        return num_nodes

    @staticmethod
    def create(batch_size=1, use_packing=False, options=None, name=None):
        """
        Static factory method for creating this factory with the root location
        customised for packing/padding. This can help save time downloading and
        processing the dataset for repeated runs.
        """
        suffix = "packed" if use_packing else "padded"

        return HydroNetDatasetFactory(
            root=dataroot(f"hydronet_{suffix}"),
            batch_size=batch_size,
            use_packing=use_packing,
            options=options,
            name=name,
        )


class LazyDataset(torch.utils.data.Dataset):
    """
    Lazy initialized dataset.

    A lightweight decorator for lazy-initializing a map-style dataset. This
    approach can save the cost of serializing & deserializing the entire
    dataset when using asynchronous data loading.

    This class is intended to be used in combination with the DatasetFactory
    and is not intended to be called directly. Performance critical methods of
    this class are instrumented for profiling with PopVision System Analyser.
    """

    def __init__(self, dataset_factory: DatasetFactory, split: str = "train"):
        """
        :param dataset_factory (DatasetFactory): The dataset factory instance
        :param split (str): The desired dataset split to load.
            (default: "train")
        """
        super().__init__()
        self._dataset = None
        self._factory = dataset_factory
        assert split in (
            "train",
            "val",
            "test",
        ), f"Invalid split = {split}. Must be 'train', 'val', or 'test'."
        self._attr = split + "_dataset"

    def _reset(self):
        """
        resets the dataset so that both this class and the factory instance are
        not holding a reference to a materialized dataset.
        """
        channel = poptorch.profiling.Channel(self.__class__.__name__)
        with channel.tracepoint("_reset"):
            self._dataset = None
            self._factory.reset()

    def __len__(self):
        """
        Dataset length

        This method is called on the main process so we load the dataset, get
        the length, and then reset to avoid the cost of serializing the entire
        dataset for asynchronous workers.
        """
        channel = poptorch.profiling.Channel(self.__class__.__name__)
        with channel.tracepoint("__len__"):
            self._load()
            L = len(self._dataset)
            self._reset()
            return L

    def _load(self):
        """
        Loads the specified dataset split using the factory.
        """
        if self._dataset is None:
            channel = poptorch.profiling.Channel(self.__class__.__name__)
            with channel.tracepoint("load"):
                self._dataset = getattr(self._factory, self._attr)

    def __getitem__(self, idx):
        """
        Get an example from the map-style dataset

        Lazy initializes the dataset on asynchronous workers.
        """
        channel = poptorch.profiling.Channel(self.__class__.__name__)
        with channel.tracepoint("__getitem__"):
            self._load()
            return self._dataset.__getitem__(idx)

    @property
    def strategy(self):
        """
        Forwards the packing strategy property to the dataset factory.
        """
        return self._factory.strategy
