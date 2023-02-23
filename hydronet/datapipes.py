# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import time

import poptorch
import torch
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data.datapipes.iter import Batcher
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import QM9
from tqdm import tqdm


def select_keys(data):
    return Data(z=data.z, pos=data.pos, y=data.y[0, 4].view(-1))


def combine(batches):
    num_outputs = len(batches[0])
    outputs = [None] * num_outputs

    for i in range(num_outputs):
        outputs[i] = torch.stack([item[i] for item in batches])

    return tuple(outputs)


def combined_batching(pipe, options=poptorch.Options()):
    num_mini_batches = (
        options.replication_factor
        * options.device_iterations
        * options.Training.gradient_accumulation
    )

    pipe = pipe.batch(batch_size=num_mini_batches, drop_last=True)
    return pipe.map(combine)


def padding_graph(num_nodes):
    return Data(
        z=torch.zeros(num_nodes, dtype=torch.long),
        pos=torch.zeros(num_nodes, 3),
        y=torch.zeros(1),
    )


class FixedNodeBatch:
    def __init__(self, max_num_nodes):
        self.max_num_nodes = max_num_nodes

    def __call__(self, items: list):
        num_nodes = sum(g.num_nodes for g in items)
        num_pad = self.max_num_nodes - num_nodes
        items.append(padding_graph(num_pad))
        return Batch.from_data_list(items)


@functional_datapipe("fixed_size_batch")
class FixedSizeBatcher(Batcher):
    def __init__(self, datapipe, batch_size: int, max_nodes_per_graph: int):
        max_num_nodes = (batch_size - 1) * max_nodes_per_graph
        super().__init__(
            datapipe,
            batch_size - 1,
            drop_last=True,
            wrapper_class=FixedNodeBatch(max_num_nodes=max_num_nodes),
        )


@functional_datapipe("to_tuple")
class ToTuple(IterDataPipe):
    def __init__(self, dp, include_keys) -> None:
        super().__init__()
        self.dp = dp
        self.include_keys = include_keys

    def __iter__(self):
        for item in self.dp:
            yield tuple(item[k] for k in self.include_keys)

    def __len__(self):
        return len(self.dp)


def noop(*args):
    pass


def create_dataloader(pipe):
    num_workers = 16
    async_options = {
        "sharing_strategy": poptorch.SharingStrategy.SharedMemory,
        "early_preload": True,
        "buffer_size": 2,
        "load_indefinitely": True,
        "miss_sleep_time_in_ms": 0,
    }

    return poptorch.DataLoader(
        poptorch.Options(),
        pipe,
        mode=poptorch.DataLoaderMode.Async,
        num_workers=num_workers,
        persistent_workers=True,
        async_options=async_options,
        prefetch_factor=2,
        worker_init_fn=noop,
    )


def create_nativeloader(pipe):
    from torch.utils.data import DataLoader

    num_workers = 16

    return DataLoader(
        pipe,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )


if __name__ == "__main__":
    options = poptorch.Options()
    options.replicationFactor(16)
    options.deviceIterations(32)
    options.Training.gradientAccumulation(1)

    dp = QM9.to_datapipe(root="data/QM9", transform=select_keys)
    dp = dp.fixed_size_batch(batch_size=8, max_nodes_per_graph=32)
    dp = dp.to_tuple(["z", "pos", "batch", "y"])
    dp = combined_batching(dp, options)

    tic = time.perf_counter()
    # loader = create_dataloader(dp)
    loader = create_nativeloader(dp)
    print(time.perf_counter() - tic)
    N = 0
    for batch in tqdm(loader):
        N += int(torch.count_nonzero(batch[-1]))

    print(N, len(QM9(root="data/QM9", transform=select_keys)))

    for _ in range(4):
        tic = time.perf_counter()
        for batch in tqdm(loader):
            pass
        print(time.perf_counter() - tic)
