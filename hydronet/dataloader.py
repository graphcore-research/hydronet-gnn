# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from itertools import chain
from typing import List, Optional, Tuple, Union

import poptorch
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data, Dataset

from .data_utils import data_keys
from .packing import PackingStrategy


class TupleCollater(object):
    """
    Collate a PyG Batch as a tuple of tensors
    """

    def __init__(self, include_keys: Optional[Union[List[str], Tuple[str]]] = None):
        """
        :param include_keys (optional): Keys to include from the batch in the
            output tuple specified as either a list or tuple of strings. The
            ordering of the keys is preserved in the tuple. By default will
            extract the keys required for training the SchNet model.
        """
        super().__init__()
        self.include_keys = include_keys

        if self.include_keys is None:
            # Use the defaults + the "ptr" vector with the same ordering as
            # the forward method of the SchNet model.
            keys = list(data_keys())
            keys.insert(2, "ptr")
            self.include_keys = tuple(keys)

        channel = poptorch.profiling.Channel(self.__class__.__name__)
        channel.instrument(self, "__call__")

    def __call__(self, data_list):
        with poptorch.profiling.Channel("Batch").tracepoint("from_data_list"):
            batch = Batch.from_data_list(data_list)
            ptr = batch.ptr[1:-1]
            graph_pad = max(batch.y.shape[0] - ptr.shape[0], 0)
            ptr = F.pad(ptr, (0, graph_pad))
            batch.ptr = ptr.to(torch.int8)

        assert all(
            [hasattr(batch, k) for k in self.include_keys]
        ), f"Batch is missing a required key: '{self.include_keys}'"

        return tuple(getattr(batch, k) for k in self.include_keys)


class CombinedBatchingCollater(object):
    """Collator object that manages the combined batch size defined as:

        combined_batch_size = mini_batch_size * device_iterations
                             * replication_factor * gradient_accumulation

    This is intended to be used in combination with the poptorch.DataLoader
    """

    def __init__(
        self,
        mini_batch_size: int,
        strategy: Optional[PackingStrategy] = None,
        k: Optional[int] = None,
    ):
        """
        :param mini_batch_size (int): mini batch size used by the SchNet model
        :param strategy (PackingStrategy): The packing strategy used
        :param k (int): The number of nearest neighbors used in building the
            graphs.
        """
        super().__init__()
        self.mini_batch_size = mini_batch_size
        self.strategy = strategy
        self.k = k
        self.tuple_collate = TupleCollater()
        channel = poptorch.profiling.Channel(self.__class__.__name__)
        channel.instrument(self, "_prepare_package", "__call__")

    def _prepare_package(self, packages):
        num_packages = len(packages)
        total_num_graphs = num_packages * self.strategy.max_num_graphs
        total_num_nodes = num_packages * self.strategy.max_num_nodes
        graphs = list(chain.from_iterable(packages))

        num_nodes = 0
        for i, g in enumerate(graphs):
            num_nodes += g.num_nodes
            g.y = g.y.view(-1)
            graphs[i] = g

        num_filler_nodes = total_num_nodes - num_nodes

        if num_filler_nodes > 0:
            graphs.append(create_packing_molecule(num_filler_nodes))

        num_graphs = len(graphs)

        if num_graphs < total_num_graphs:
            last = graphs[-1]
            last.y = F.pad(last.y, (0, total_num_graphs - num_graphs))
            graphs[-1] = last

        return graphs

    def __call__(self, batch):
        num_items = len(batch)
        assert num_items % self.mini_batch_size == 0, (
            "Invalid batch size. "
            f"{num_items} graphs and mini_batch_size={self.mini_batch_size}."
        )

        num_mini_batches = num_items // self.mini_batch_size
        batches = [None] * num_mini_batches
        start = 0
        stride = self.mini_batch_size

        for i in range(num_mini_batches):
            slices = batch[start : start + stride]

            if self.strategy is not None:
                slices = self._prepare_package(slices)

            batches[i] = self.tuple_collate(slices)
            start += stride

        num_outputs = len(batches[0])
        outputs = [None] * num_outputs

        for i in range(num_outputs):
            outputs[i] = torch.cat(tuple(item[i] for item in batches))

        return tuple(outputs)


class PackingCollater(object):
    """
    Collater object for packed datasets

    This collater is expected to be used along with the k-nearest neighbors
    method for determining the graph edges.
    """

    def __init__(
        self,
        package_batch_size: int,
        strategy: PackingStrategy,
        k: int,
        debug: bool = False,
    ):
        """
        :param package_batch_size (int): The number of packages to be processed
            in a single step.
        :param strategy (PackingStrategy): The packing strategy used to pack
            the dataset.
        :param k (int): The number of nearest neighbors used in building the
            graphs.
        """
        super().__init__()
        self.package_batch_size = package_batch_size
        self.strategy = strategy
        self.k = k
        self.debug = debug
        self.collater = CombinedBatchingCollater(
            self.package_batch_size, strategy=strategy, k=k
        )

        channel = poptorch.profiling.Channel(self.__class__.__name__)
        channel.instrument(self, "_prepare_package", "__call__")

    def _prepare_package(self, package):
        """Prepares each package by padding any incomplete data tensors"""
        num_nodes = 0

        for i, graph in enumerate(package):
            num_nodes += graph.num_nodes
            graph.y = graph.y.view(-1)
            package[i] = graph

        assert num_nodes <= self.strategy.max_num_nodes, (
            f"Too many nodes in package. Package contains {num_nodes} nodes "
            f"and maximum is {self.strategy.max_num_nodes}."
        )

        num_graphs = len(package)
        max_num_graphs = self.strategy.max_num_graphs
        assert num_graphs <= max_num_graphs, (
            f"Too many graphs in package. Package contains {num_graphs} "
            f"graphs and maximum is {max_num_graphs}."
        )

        return package

    def __call__(self, data_list):
        num_packages = len(data_list)
        assert num_packages % self.package_batch_size == 0

        if self.debug:
            data_list = [self._prepare_package(p) for p in data_list]

        return self.collater(data_list)


def create_dataloader(
    dataset: Dataset,
    ipu_opts: Optional[poptorch.Options] = None,
    use_packing: bool = False,
    batch_size: int = 1,
    shuffle: bool = False,
    k: Optional[int] = None,
    num_workers: int = 0,
    buffer_size: Optional[int] = None,
    use_async_loader: bool = True,
):
    """
    Creates a data loader for graph datasets

    Applies the mini-batching method of concatenating multiple graphs into a
    single graph with multiple disconnected subgraphs. See:

    https://pytorch-geometric.readthedocs.io/en/2.0.2/notes/batching.html
    """
    if ipu_opts is None:
        ipu_opts = poptorch.Options()

    if use_packing:
        collater = PackingCollater(batch_size, dataset.strategy, k)
    else:
        collater = CombinedBatchingCollater(batch_size)

    buffer_size = 3 if buffer_size is None else buffer_size

    async_options = {
        "sharing_strategy": poptorch.SharingStrategy.ForkServer,
        "early_preload": True,
        "buffer_size": buffer_size,
        "load_indefinitely": True,
        "miss_sleep_time_in_ms": 0,
    }

    if use_async_loader and num_workers > 0:
        mode = poptorch.DataLoaderMode.Async
    else:
        mode = poptorch.DataLoaderMode.Sync

    return poptorch.DataLoader(
        ipu_opts,
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        mode=mode,
        async_options=async_options,
        collate_fn=collater,
    )


def create_packing_molecule(num_atoms: int):
    """
    Creates a packing molecule

    A non-interacting molecule that is used to fill incomplete batches when
    using packed datasets.

    :param num_atoms (int): The number of atoms in the packing molecule.

    :returns: Data instance
    """
    z = torch.zeros(num_atoms, dtype=torch.int8)
    pos = torch.zeros(num_atoms, 3, dtype=torch.half)
    y = torch.zeros(1)
    return Data(z=z, pos=pos, y=y, num_nodes=num_atoms)
