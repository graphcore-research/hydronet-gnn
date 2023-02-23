# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import copy
from functools import cached_property
from itertools import chain
from typing import Optional

import torch

from .lpfhp import pack_using_lpfhp


class PackingStrategy(object):
    """Graph packing strategy.

    Each package is comprised of at least one graph to satisfy the constraint
    of fitting within a maximum total number of nodes. Concretely, a package is
    defined by:

        * Package: [na, nb, nc, ...]
        * Count: C

    where a package is constrained to satisfy:

        sum(na, nb, nc, ...)  <= max_num_nodes

    The count C encodes the number of packages with contents [na, nb, nc, ...]
    that can be assembled from across the entire dataset. The PackingStrategy
    for a dataset is described by:

        * sequence: ([na, nb, ...], [nc, nd, ...], ...)
        * counts: (U, V, ...)
        * max_num_nodes: N

    The sequence and counts are computed by analyzing the histogram of the
    number of nodes extracted from a graph dataset.

    Please refer to "Packing: Towards 2x NLP BERT Acceleration" by M Kosec,
    S Fu, and M Krell for further details of the algorithms used to compute the
    packing strategy <https://arxiv.org/abs/2107.02027>.
    """

    def __init__(self, sequence: tuple, counts: tuple, max_num_nodes: int):
        """
        :param sequence (tuple): The sequence of graph packages as a tuple of
            lists of ints: ([na, nb, ...], ...)
        :param counts (tuple): The number of packages that can be assembled
            from the dataset as a tuple of ints: (U, V, ...)
        :param max_num_nodes (int): The maximum total number of nodes for a
            single package of graphs.
        """
        super().__init__()
        self.sequence = sequence
        self.counts = counts
        self.max_num_nodes = max_num_nodes

        # Validate that the inputs are consistent with requirements.
        assert isinstance(self.sequence, tuple), "Sequence must be a tuple."

        for package in self.sequence:
            is_list = isinstance(package, list)
            isvalid = is_list and all(isinstance(n, int) for n in package)
            assert isvalid, "Each package must be a list of ints."

        assert isinstance(self.counts, tuple), "Counts must be a tuple."
        assert all(
            isinstance(c, int) for c in self.counts
        ), "Each count must be an int."

        assert len(self.sequence) == len(
            self.counts
        ), "Sequence length must match counts length"

        pack_lengths = [sum(s) for s in self.sequence]
        valid_lengths = all(ln <= self.max_num_nodes for ln in pack_lengths)
        assert (
            valid_lengths
        ), f"Found package that exceeds max of {self.max_num_nodes} nodes."

    @cached_property
    def packed_len(self):
        """The length of the packed dataset"""
        return sum(self.counts)

    @cached_property
    def unpacked_len(self):
        """The length of the unpacked dataset"""
        total = 0
        for s, c in zip(self.sequence, self.counts):
            total += len(s) * c

        return total

    @cached_property
    def max_num_graphs(self):
        """The maximum number of graphs that will be packed together"""
        num_graphs = []
        num_nodes = []
        for package in self.sequence:
            num_graphs.append(len(package))
            num_nodes.append(sum(package))

        num_graphs = torch.tensor(num_graphs)
        num_nodes = torch.tensor(num_nodes)

        # increment incomplete packages to account for the filler graph
        num_graphs[num_nodes < self.max_num_nodes] += 1
        return num_graphs.max().item()

    @cached_property
    def node_efficiency(self):
        total_nodes = self.packed_len * self.max_num_nodes
        num_padding_nodes = 0
        for s, c in zip(self.sequence, self.counts):
            num_padding_nodes += c * (self.max_num_nodes - sum(s))

        return 100 - num_padding_nodes / total_nodes * 100

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}("
        s += f"packed_len: {self.packed_len}, "
        s += f"unpacked_len: {self.unpacked_len}, "
        s += f"max_num_nodes: {self.max_num_nodes}, "
        s += f"max_num_graphs: {self.max_num_graphs}, "
        s += f"node_efficiency: {self.node_efficiency:.1f})"
        return s

    @staticmethod
    def from_histogram(
        histogram: torch.Tensor, max_num_nodes: int, max_num_graphs: int
    ):
        extension = histogram.new_zeros(max_num_nodes - histogram.numel())
        extension = torch.cat([histogram, extension])
        result = pack_using_lpfhp(extension.tolist(), max_num_nodes, max_num_graphs)
        sequence, counts = result
        return PackingStrategy(tuple(sequence), tuple(counts.tolist()), max_num_nodes)


class Packer(object):
    """
    Helper class for packing a graph dataset based on a strategy
    """

    def __init__(
        self, strategy: PackingStrategy, num_nodes: torch.Tensor, shuffle: bool = False
    ):
        """
        :param strategy (PackingStrategy): The pre-computed strategy.
        :param num_nodes (torch.Tensor): A vector containing the number of
            nodes for every graph in the dataset.
        :param shuffle (bool): Randomly shuffle the dataset (default: False)
        """
        super().__init__()
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.shuffle = shuffle
        assert self.num_nodes.numel() == self.strategy.unpacked_len, (
            f"The number of graphs {self.num_nodes.numel()} does not match "
            f"the PackingStrategy {self.strategy.unpacked_len}."
        )

        self._create_mappings()

    def _create_mappings(self):
        """Private method for initializing mappings needed for graph packing"""
        unique_num_nodes = set()
        for i in self.strategy.sequence:
            unique_num_nodes.update(i)

        indices_map = dict()
        slice_map = dict()
        for n in unique_num_nodes:
            indices = torch.nonzero(self.num_nodes == n)

            if self.shuffle:
                perm = torch.randperm(indices.numel())
                indices = indices[perm]

            indices = indices.flatten()
            indices_map[n] = indices.tolist()
            slice_map[n] = 0

        # Mapping from num_nodes -> indices in dataset
        self.indices_map = indices_map

        # Mapping from num_nodes -> current slice in the indices list
        self.slice_map = slice_map

    def _fill_package(self, package, num_packages):
        """Private method for filling a single package"""
        out = []
        for num_nodes in package:
            indices = self.indices_map[num_nodes]
            start = self.slice_map[num_nodes]
            end = start + num_packages

            assert end <= len(indices), (
                f"Not enough graphs with {num_nodes} nodes to "
                f"fulfill the requested package: {package}, {num_packages}."
            )

            indices = indices[start:end]
            out.append(indices)
            self.slice_map[num_nodes] = end

        return tuple(zip(*out))

    def pack_all(self):
        """
        Packs the dataset according to the supplied strategy.

        :returns (tuple): Tuple of indices to select from the unpacked dataset.
        """
        out = []

        for package, num_packages in zip(self.strategy.sequence, self.strategy.counts):
            out.append(self._fill_package(package, num_packages))

        out = tuple(chain.from_iterable(out))

        if not self.shuffle:
            return out

        perm = torch.randperm(len(out))
        perm = perm.tolist()
        return tuple(out[idx] for idx in perm)


class PackedDataset(torch.utils.data.Dataset):
    """
    Wrapper class for packing a dataset.
    """

    def __init__(
        self,
        unpacked_dataset: torch.utils.data.Dataset,
        strategy: PackingStrategy,
        num_nodes: Optional[torch.Tensor] = None,
        shuffle: bool = False,
    ):
        """
        :param (torch.utils.data.Dataset): The unpacked map-style graph dataset
        :param strategy (PackingStrategy): Strategy to use for packing the
            dataset
        :param num_nodes (Tensor, optional): Tensor containing the number of
            nodes for each graph in the dataset. This input is optional and
            will be computed by iterating over the dataset when omitted.
        :param shuffle (bool): Randomly shuffle the dataset (default: False)
        """
        super().__init__()
        self.unpacked_dataset = unpacked_dataset
        self.strategy = strategy
        self.num_nodes = num_nodes

        if self.num_nodes is None:
            self.num_nodes = torch.tensor([d.num_nodes for d in self.unpacked_dataset])

        packer = Packer(self.strategy, self.num_nodes, shuffle)
        self.packages = packer.pack_all()

    def __len__(self):
        return len(self.packages)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            p = self.packages[idx]
            return [self.unpacked_dataset[i] for i in p]

        assert isinstance(idx, slice), f"Expected slice object not {type(idx)}."

        subset = copy.copy(self)
        subset.packages = self.packages[idx]
        return subset


def qm9_packing_strategy():
    """
    Packing strategy for the QM9 molecular dataset.
    """
    # yapf: disable
    sequence = ([18, 18], [17, 19], [16, 20], [15, 21], [14, 22], [13, 23],
                [12, 24], [11, 25], [10, 26], [9, 27], [7, 29], [4, 5, 27],
                [4, 16, 16], [27], [25], [23], [21], [16, 19], [14, 21],
                [12, 23], [10, 25], [8, 27], [6, 29], [3, 16, 16],
                [16, 16], [5, 29])

    counts = (8721, 16969, 12403, 10216, 4428, 4027, 712, 1053, 59, 172, 20, 2,
              2, 117, 446, 858, 620, 1177, 2330, 1477, 424, 65, 12, 2, 118, 3)
    # yapf: enable

    max_num_nodes = 36
    return PackingStrategy(sequence, counts, max_num_nodes)


def hydronet_packing_strategy():
    """ """
    max_num_nodes = 154
    max_num_graphs = 5
    histogram = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            10,
            0,
            0,
            19,
            0,
            0,
            106,
            0,
            0,
            470,
            0,
            0,
            577,
            0,
            0,
            4126,
            0,
            0,
            11448,
            0,
            0,
            26824,
            0,
            0,
            42641,
            0,
            0,
            58452,
            0,
            0,
            80533,
            0,
            0,
            95820,
            0,
            0,
            96859,
            0,
            0,
            106987,
            0,
            0,
            112679,
            0,
            0,
            106808,
            0,
            0,
            121376,
            0,
            0,
            314062,
            0,
            0,
            394711,
            0,
            0,
            353964,
            0,
            0,
            315329,
            0,
            0,
            482907,
            0,
            0,
            530503,
            0,
            0,
            392145,
            0,
            0,
            276037,
            0,
            0,
            295831,
            0,
            0,
            243514,
            0,
        ]
    )
    return PackingStrategy.from_histogram(histogram, max_num_nodes, max_num_graphs)


def hydronet_medium_packing_strategy():
    max_num_nodes = 192
    max_num_graphs = 4
    histogram = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            10,
            0,
            0,
            19,
            0,
            0,
            106,
            0,
            0,
            470,
            0,
            0,
            577,
            0,
            0,
            4126,
            0,
            0,
            11448,
            0,
            0,
            26824,
            0,
            0,
            42641,
            0,
            0,
            58452,
            0,
            0,
            80533,
            0,
            0,
            95820,
            0,
            0,
            96859,
            0,
            0,
            106987,
            0,
            0,
            112679,
            0,
            0,
            106808,
            0,
            0,
            121376,
            0,
            0,
            314062,
            0,
            0,
            394711,
            0,
            0,
            353964,
            0,
            0,
            315329,
            0,
            0,
            482907,
            0,
        ]
    )
    return PackingStrategy.from_histogram(histogram, max_num_nodes, max_num_graphs)


def hydronet_small_packing_strategy():
    max_num_nodes = 192
    max_num_graphs = 5
    histogram = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            5,
            0,
            0,
            23,
            0,
            0,
            87,
            0,
            0,
            97,
            0,
            0,
            763,
            0,
            0,
            2106,
            0,
            0,
            4917,
            0,
            0,
            7967,
            0,
            0,
            10704,
            0,
            0,
            14935,
            0,
            0,
            17822,
            0,
            0,
            17662,
            0,
            0,
            19534,
            0,
            0,
            20736,
            0,
            0,
            19489,
            0,
            0,
            22139,
            0,
            0,
            57318,
            0,
            0,
            72693,
            0,
            0,
            64525,
            0,
            0,
            57671,
            0,
            0,
            88806,
        ]
    )
    return PackingStrategy.from_histogram(histogram, max_num_nodes, max_num_graphs)
