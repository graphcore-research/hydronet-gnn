# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import time
from itertools import chain

import pytest
import torch
from poptorch import Options
from tqdm import tqdm

from hydronet.dataloader import create_dataloader
from hydronet.packing import (
    PackedDataset,
    Packer,
    PackingStrategy,
    qm9_packing_strategy,
)
from hydronet.transforms import create_transform


@pytest.fixture
def example_strategy():
    sequence = ([8], [1, 7], [2, 6], [3, 5], [4, 4], [2, 2, 2, 2])
    counts = (1, 2, 4, 2, 1, 1)
    max_num_nodes = 8

    return PackingStrategy(sequence, counts, max_num_nodes)


@pytest.fixture
def example_num_nodes(example_strategy):
    num_nodes = []

    for sequence, count in zip(example_strategy.sequence, example_strategy.counts):
        new_nodes = ([n] * count for n in sequence)
        num_nodes += chain.from_iterable(new_nodes)

    torch.manual_seed(0)
    perm = torch.randperm(len(num_nodes))
    num_nodes = torch.tensor(num_nodes)
    return num_nodes[perm]


def test_packing_strategy(example_strategy):
    assert example_strategy.unpacked_len == 23
    assert example_strategy.packed_len == 11
    assert example_strategy.max_num_nodes == 8
    assert example_strategy.max_num_graphs == 4


@pytest.mark.parametrize("shuffle", [True, False])
def test_graph_packer(example_strategy, example_num_nodes, shuffle):
    if shuffle:
        torch.manual_seed(17)

    packer = Packer(example_strategy, example_num_nodes, shuffle)
    packages = packer.pack_all()

    assert len(packages) == example_strategy.packed_len

    for p in packages:
        total_packed_nodes = sum(example_num_nodes[i] for i in p)
        assert total_packed_nodes <= example_strategy.max_num_nodes


@pytest.mark.parametrize("shuffle", [True, False])
def test_qm9_packing_strategy(qm9_num_nodes, shuffle):
    if shuffle:
        torch.manual_seed(17)

    strategy = qm9_packing_strategy()
    packer = Packer(strategy, qm9_num_nodes, shuffle)
    packages = packer.pack_all()

    assert len(packages) == strategy.packed_len

    for p in packages:
        total_packed_nodes = sum(qm9_num_nodes[i] for i in p)
        assert total_packed_nodes <= strategy.max_num_nodes


@pytest.mark.parametrize("shuffle", [True, False])
def test_qm9_packed_dataset(pyg_qm9, qm9_num_nodes, shuffle):
    if shuffle:
        torch.manual_seed(17)

    strategy = qm9_packing_strategy()
    packed_dataset = PackedDataset(
        unpacked_dataset=pyg_qm9,
        strategy=strategy,
        num_nodes=qm9_num_nodes,
        shuffle=shuffle,
    )

    assert len(packed_dataset) == strategy.packed_len

    for package in tqdm(packed_dataset):
        assert sum(d.num_nodes for d in package) <= strategy.max_num_nodes


@pytest.mark.requires_ipu
@pytest.mark.parametrize("shuffle", [False, True])
def test_qm9_packed_dataloader(pyg_qm9, qm9_num_nodes, shuffle):
    strategy = qm9_packing_strategy()
    k = 8
    pyg_qm9.transform = create_transform(
        use_padding=False,
        use_qm9_energy=True,
        use_standardized_energy=True,
    )

    batch_size = 32
    device_iterations = 4
    replication_factor = 4
    options = Options().deviceIterations(device_iterations)
    options.replicationFactor(replication_factor)

    dataset = PackedDataset(
        pyg_qm9, strategy=strategy, num_nodes=qm9_num_nodes, shuffle=shuffle
    )

    tic = time.perf_counter()
    loader = create_dataloader(
        dataset=dataset,
        ipu_opts=options,
        use_packing=True,
        batch_size=batch_size,
        num_workers=1,
        k=k,
    )
    elapsed = time.perf_counter() - tic
    print(f"Time to create data loader: {elapsed:0.02f} s")

    num_nodes_per_batch = loader.combinedBatchSize * strategy.max_num_nodes
    num_mols_per_batch = loader.combinedBatchSize * strategy.max_num_graphs
    total_graphs = 0
    for package in tqdm(loader):
        z, pos, ptr, y = package

        assert z.numel() == num_nodes_per_batch
        assert pos.shape == (num_nodes_per_batch, 3)
        assert ptr.numel() == num_mols_per_batch
        assert y.numel() == num_mols_per_batch
        total_graphs += (y != 0.0).sum()

    assert total_graphs <= strategy.unpacked_len

    num_runs = 5
    for _ in range(num_runs):
        tic = time.perf_counter()
        for batch in loader:
            _
        elapsed = time.perf_counter() - tic
        print(f"Time: {elapsed:0.02f}s")
        print(f"Throughput: {strategy.unpacked_len/elapsed:.1f} molecules/s")

    loader.terminate()
