# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import enum
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import pandas as pd
import poptorch
import torch
from jsonargparse import CLI
from torch_geometric.nn.models.schnet import InteractionBlock
from tqdm import tqdm

from .dataset_factory import HydroNetDatasetFactory, QM9DatasetFactory
from .model import SchNet

try:
    import wandb
except ImportError:
    wandb = None

torch.set_num_threads(1)


class Timer:
    def __enter__(self):
        self.tic = time.perf_counter()

    def __exit__(self, type, value, traceback):
        self.elapsed = time.perf_counter() - self.tic


class Logger:
    def __init__(self, args):
        self.use_wandb = args["use_wandb"]
        self.epochs = []

        if self.use_wandb:
            wandb.init(
                project="hydronet-gnn",
                settings=wandb.Settings(console="wrap"),
                config=args,
            )

    def model(self, model):
        hparams = model.hyperparameters()
        if self.use_wandb:
            wandb.config.update(hparams, allow_val_change=True)

    def log(self, data, commit: Optional[bool] = None):
        if self.use_wandb:
            wandb.log(data, commit=commit)

    def epoch(self, data):
        self.epochs.append(data)

    def summarize_epochs(self, times):
        assert len(times) == len(self.epochs)
        rows = []
        for tt, data in zip(enumerate(times), self.epochs):
            d = {}
            d["train/epoch"] = tt[0]
            d["train/parent_time"] = tt[1]
            d["train/self_time"] = data["time"]
            d["train/loss_min"] = float(data["loss"].min())
            d["train/loss_max"] = float(data["loss"].max())
            d["train/loss_mean"] = float(data["loss"].mean())
            d["N"] = data["N"]
            if self.use_wandb:
                wandb.log(d)
            rows.append(d)

        df = pd.DataFrame(rows)
        print(df.describe())
        self.epochs = []


@dataclass
class CombinedBatch:
    model_batch_size: int
    device_iterations: int = 1
    replication_factor: int = 1
    gradient_accumulation: int = 1

    @property
    def global_batch_size(self):
        return (
            self.model_batch_size
            * self.device_iterations
            * self.replication_factor
            * self.gradient_accumulation
        )


@dataclass
class ModelParams:
    num_features: int = 100
    num_interactions: int = 4
    num_gaussians: int = 25
    cutoff: float = 6.0


def poptorch_options(
    synthetic_data,
    available_memory_proportion,
    combined_batch,
    profile_dir,
    prefetch_depth,
    num_io_tiles,
    merge_all_reduce,
):
    options = poptorch.Options()

    if available_memory_proportion is not None:
        amp_dict = {
            f"IPU{i}": available_memory_proportion
            for i in range(combined_batch.replication_factor)
        }
        options.setAvailableMemoryProportion(amp_dict)

    options.deviceIterations(combined_batch.device_iterations)
    options.replicationFactor(combined_batch.replication_factor)
    options.enableSyntheticData(synthetic_data)
    options.Training.gradientAccumulation(combined_batch.gradient_accumulation)

    if prefetch_depth > 0:
        options._Popart.set("defaultBufferingDepth", prefetch_depth)

    if num_io_tiles is not None and num_io_tiles > 0:
        options.outputMode(poptorch.OutputMode.All)
        options.TensorLocations.numIOTiles(num_io_tiles)
        options.setExecutionStrategy(poptorch.ShardedExecution())

    if profile_dir:
        options.enableProfiling(profile_dir)

    if not merge_all_reduce:
        return options

    # AccumulateOuterFragmentSchedule.OverlapCycleOptimized: 2
    options._Popart.set("accumulateOuterFragmentSettings.schedule", 2)
    options._Popart.set(
        "replicatedCollectivesSettings.prepareScheduleForMergingCollectives", True
    )
    options._Popart.set("replicatedCollectivesSettings.mergeAllReduceCollectives", True)

    return options


def create_module(module, model_params, batch_size, overlap_io, k):
    if module == BenchModule.interaction:
        block = InteractionBlock(
            hidden_channels=model_params.num_features,
            num_gaussians=model_params.num_gaussians,
            num_filters=model_params.num_features,
            cutoff=model_params.cutoff,
        )

        block.hyperparameters = lambda: asdict(module)
        return block
    if module == BenchModule.schnet:
        factory_fn = SchNet
    else:
        factory_fn = SchNet.create_with_fss

    model = factory_fn(
        batch_size=batch_size,
        num_interactions=model_params.num_interactions,
        num_features=model_params.num_features,
        overlap_io=overlap_io,
        k=k,
    )
    model.train()
    return model


def synthetic_epoch(model, loader, data, logger):
    num_steps = len(loader)
    timer = Timer()

    with timer:
        for _ in tqdm(range(num_steps)):
            _, _ = model(*data)

    logger.epoch(
        {"N": num_steps, "loss": torch.zeros(num_steps), "time": timer.elapsed}
    )


def epoch(model, loader, logger):
    N = 0
    losses = []
    timer = Timer()

    with timer:
        for data in tqdm(loader):
            _, loss = model(*data)
            losses.append(loss)
            N += (data[-1] != 0).sum().item()

    logger.epoch({"N": N, "loss": torch.cat(losses), "time": timer.elapsed})


class Task(enum.Enum):
    """Run the training loop"""

    train = 0
    """Run a single training step for generating a PopVision graph profile"""
    profile = 1
    """Simulate dataloading to estimate maximum host throughput"""
    dataloading = 2


class BenchDataset(enum.Enum):
    qm9 = "qm9"  # QM9 (#graphs=130831)
    hydronet = "large"  # HydroNet (#graphs=4464740)
    hydronet_medium = "medium"  # HydroNet medium (#graphs=2726710)
    hydronet_small = "small"  # HydroNet medium (#graphs=500000)


class BenchModule(enum.Enum):
    schnet = "schnet"
    schnet_fss = "schnet_fss"
    interaction = "interaction"


def create_dataset_factory(dataset, batch_size, use_packing, options):
    assert isinstance(dataset, BenchDataset)
    if dataset == BenchDataset.qm9:
        return QM9DatasetFactory.create(batch_size, use_packing, options)

    return HydroNetDatasetFactory.create(
        batch_size, use_packing, options, name=dataset.value
    )


def bench(
    dataset: BenchDataset = BenchDataset.qm9,
    use_packing: bool = True,
    synthetic_data: bool = False,
    available_memory_proportion: Optional[float] = None,
    combined_batch: CombinedBatch = CombinedBatch(
        model_batch_size=16,
        device_iterations=1,
        replication_factor=16,
        gradient_accumulation=8,
    ),
    model_params: ModelParams = ModelParams(),
    module: BenchModule = BenchModule.schnet_fss,
    learning_rate: float = 1e-3,
    num_epochs: int = 25,
    profile_dir: Optional[str] = None,
    prefetch_depth: int = 3,
    num_io_tiles: Optional[int] = 0,
    task: Task = Task.train,
    seed: int = 0,
    max_workers: int = 32,
    use_async_loader: bool = True,
    merge_all_reduce: bool = True,
    use_wandb: bool = True,
):
    """
    IPU Benchmark of SchNet model

    :param dataset: qm9, hydronet, hydronet_medium, or hydronet_small
    :param use_packing: Apply graph packing to the dataset.
    :param synthetic_data: Use synthetic data on the device to disable I/O.
    :param available_memory_proportion: the AMP budget used for planning ops.
    :param batch_size: The batch size used by the data loader.
    :param replication_factor: The number of data parallel replicas.
    :param device_iterations: The number of device iterations to use.
    :param gradient_accumulation: The number of mini-batches to accumulate for
        the gradient calculation.
    :param module: module to benchmark, schnet, schnet_fss, or interaction
    :param learning_rate: The learning rate used by the optimiser.
    :param num_epochs: The number of epochs to benchmark.
    :param profile_dir: Run a single training step with profiling enabled and
        saves the profiling report to the provided location.
    :param prefetch_depth: The depth of the buffer used for prefetching batches
        from the host.
    :param num_io_tiles: the number of tiles used with overlapped IO.
    :param task: the benchmark task to run.
    :param seed: the random seed to use.
    :param max_workers: the maximum number of data loader workers to use.
    :param use_async_loader: Enables poptorch.DataLoaderMode.Async
    :param merge_all_reduce: Enables PopART merging of all reduce collectives
    :param use_wandb: Use Weights and Biases to log benchmark results.
    """
    torch.manual_seed(seed)
    logger = Logger(locals())

    options = poptorch_options(
        synthetic_data,
        available_memory_proportion,
        combined_batch,
        profile_dir,
        prefetch_depth,
        num_io_tiles,
        merge_all_reduce,
    )

    factory = create_dataset_factory(
        dataset, combined_batch.model_batch_size, use_packing, options
    )

    channel = poptorch.profiling.Channel("dataloader")
    timer = Timer()
    with channel.tracepoint("construction"), timer:
        loader = factory.dataloader(
            split="train", max_workers=max_workers, use_async_loader=use_async_loader
        )
    logger.log({"dataloader/construction_time": timer.elapsed})

    with channel.tracepoint("iter"), timer:
        ld_iter = iter(loader)
    logger.log({"dataloader/iter_time": timer.elapsed})

    with channel.tracepoint("next"), timer:
        data = next(ld_iter)
    logger.log({"dataloader/next_time": timer.elapsed})

    for step in range(5):
        N = 0

        with channel.tracepoint(f"epoch_{step+1}"), timer:
            for data in tqdm(loader):
                energy_target = data[-1]
                N += (energy_target != 0).sum().item()

        logger.log(
            {"epoch": step, "dataloader/epoch_time": timer.elapsed, "num_processed": N}
        )

        # Simulates other work happening between epochs
        with channel.tracepoint("other"):
            time.sleep(1)

    if task == Task.dataloading:
        return

    overlap = num_io_tiles is not None and num_io_tiles > 0
    model = create_module(
        module, model_params, factory.model_batch_size, overlap, factory.k
    )
    logger.model(model)

    optimizer = poptorch.optim.Adam(model.parameters(), lr=learning_rate)
    training_model = poptorch.trainingModel(model, options, optimizer)

    with timer:
        training_model.compile(*data)

    if profile_dir:
        training_model(*data)
        return

    logger.log({"train/compile_time": timer.elapsed})

    if synthetic_data:
        epoch_fn = partial(synthetic_epoch, data=data, logger=logger)
    else:
        epoch_fn = partial(epoch, logger=logger)

    epoch_times = []

    for step in range(num_epochs):
        with channel.tracepoint(f"epoch_{step+1}"), timer:
            epoch_fn(training_model, loader)

        epoch_times.append(timer.elapsed)

    logger.summarize_epochs(epoch_times)

    if task == Task.train:
        return

    training_model.detachFromDevice()
    # Use poptorch default options (single IPU, 1 device iteration, etc)
    # Trained weights are implicitly copied to the inferenceModel instance.
    factory.options = poptorch.Options()
    model.overlap_io = False
    model.eval()
    val_model = poptorch.inferenceModel(model)
    val_loader = factory.dataloader(split="val", num_workers=8)
    bar = tqdm(val_loader, desc="Validation")
    losses = []

    for data in bar:
        args, y = data[:-1], data[-1]
        prediction = val_model(*args)
        loss = model.loss(prediction, y.view(-1))

        losses.append(loss)
        bar.set_description(f"loss: {loss.mean().item():0.4f}")

    losses = torch.stack(losses)
    mean = losses.mean().item()
    std = losses.std().item()
    min = losses.min().item()
    max = losses.max().item()
    print(f"Mean validation loss: {mean:0.4f} +/- {std:0.4f} (eV/atom)")
    print(f"               Range: [{min:0.4f}, {max:0.4f}]")


def main():
    CLI(bench)


if __name__ == "__main__":
    main()
