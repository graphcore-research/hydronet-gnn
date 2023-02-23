# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
from typing import Callable, Dict, Optional

import pandas as pd
import poptorch
import torch
from jsonargparse import CLI
from torch_geometric.nn import Aggregation
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock
from torch_scatter import scatter
from tqdm import trange

from .model import FastShiftedSoftplus

Aggregation.set_validate_args(False)

try:
    import wandb
except ImportError:
    wandb = None


class Logger:
    def __init__(self, args) -> None:
        self.use_wandb = args["use_wandb"]
        self.metrics = []

        if self.use_wandb:
            wandb.init(
                project="mpbench", settings=wandb.Settings(console="wrap"), config=args
            )

    def log(self, data):
        self.metrics.append(data)

    def commit(self, model_params):
        df = pd.DataFrame(self.metrics)
        p = {k: [v] * len(df) for k, v in model_params.items()}
        df = pd.concat([pd.DataFrame(p), df], axis=1)
        print(df.mean().to_frame().T)
        self.metrics = []

        if self.use_wandb:
            T = wandb.Table(dataframe=df)
            wandb.log({"perf_metrics": T})


@dataclass
class ModelParams:
    num_inputs: int = 4096
    num_features: int = 128
    num_outputs: int = 65536


class GatherOp(torch.nn.Module):
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        input = torch.randn(params.num_inputs, params.num_features)
        index = torch.randint(params.num_inputs, (params.num_outputs,))
        self.register_buffer("input", input)
        self.register_buffer("index", index)
        self.register_buffer("output", self(input, index, None)[-1])

    def loop_inputs(self):
        return [self.input, self.index, self.output]

    def forward(self, input, index, _):
        return input, index, input.index_select(dim=0, index=index)


class ScatterOp(torch.nn.Module):
    def __init__(self, params: ModelParams, reduce: str = "sum") -> None:
        super().__init__()
        self.scatter = partial(scatter, dim_size=params.num_outputs, reduce=reduce)
        input = torch.randn(params.num_inputs, params.num_features)
        index = torch.randint(params.num_outputs, (params.num_inputs,))
        self.register_buffer("input", input)
        self.register_buffer("index", index)
        self.register_buffer("output", self(input, index, None)[-1])

    def loop_inputs(self):
        return [self.input, self.index, self.output]

    def forward(self, input, index, _):
        return input, index, self.scatter(input, index, dim=0)


class InteractionBlockOp(torch.nn.Module):
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.block = InteractionBlock(
            hidden_channels=params.num_features,
            num_gaussians=50,
            num_filters=params.num_features,
            cutoff=6.0,
        )
        input = torch.randn(params.num_inputs, params.num_features)
        src = torch.randint(params.num_inputs, (params.num_outputs,))
        dst = torch.randint(params.num_inputs, (params.num_outputs,))
        edge_index = torch.stack([src, dst])
        edge_weight = torch.empty(params.num_outputs).uniform_(1.0, 6.0)

        grbf = GaussianSmearing(start=0.0, stop=6.0, num_gaussians=50)
        edge_attr = grbf(edge_weight)

        self.register_buffer("input", input)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.register_buffer("edge_attr", edge_attr)
        self.register_buffer(
            "output", self(input, edge_index, edge_weight, edge_attr, None)[-1]
        )

    def loop_inputs(self):
        return [
            self.input,
            self.edge_index,
            self.edge_weight,
            self.edge_attr,
            self.output,
        ]

    def forward(self, *args):
        args = args[0:-1]
        out = self.block(*args)
        return (*args, out)


def create_options(synthetic_data, available_memory_proportion, profile_dir, cache_dir):
    options = poptorch.Options()
    options.enableSyntheticData(synthetic_data)
    options.logCycleCount(True)
    options.enableExecutableCaching(cache_dir)
    options.connectionType(poptorch.ConnectionType.OnDemand)

    if available_memory_proportion is not None:
        amp_dict = {"IPU0": available_memory_proportion}
        options.setAvailableMemoryProportion(amp_dict)

    if profile_dir:
        options.enableProfiling(profile_dir)
    return options


def create_model(
    operator: str,
    num_repeats: int,
    model_params: ModelParams,
    options: poptorch.Options,
):
    block = None
    if operator == "gather":
        block = GatherOp(model_params)

    if operator.startswith("scatter"):
        reduce = operator[operator.find("_") + 1 :]
        block = ScatterOp(model_params, reduce)

    if operator == "interaction_block":
        block = InteractionBlockOp(model_params)
        FastShiftedSoftplus.replace_activation(block)

    if block is None:
        raise ValueError(f"Unsupported operator: {operator}")

    try:
        model = Bench(num_repeats=num_repeats, operator=block)
        pop_model = poptorch.inferenceModel(model, options=options)
        pop_model.compile()
        return pop_model

    except poptorch.Error:
        return None


class Bench(torch.nn.Module):
    def __init__(self, operator: Callable, num_repeats: int) -> None:
        super().__init__()
        self.num_repeats = num_repeats
        self.operator = operator

    def forward(self):
        out = poptorch.for_loop(
            self.num_repeats, self.operator, self.operator.loop_inputs()
        )[-1]
        return torch.sum(out)


class PerfMetrics:
    r"""Track performance metrics from:
        * recorded number of cycles
        * sizes of input / output
    Assumes Mk2 Bow clock speed of 1850 MHz

    Defines an effective bandwidth from the size of the output result.
    """
    bow_clock = 1850.0  # MHz

    def __init__(self, pop_model, num_repeats) -> None:
        output = pop_model.operator.output
        numels = output.numel()
        numbytes = torch.finfo(output.dtype).bits // 8
        self.out_gib = numels * numbytes / 1024**3
        self.num_repeats = num_repeats

    def update(self, cycles):
        avg_cycles = cycles / self.num_repeats
        time_us = avg_cycles / self.bow_clock
        time_s = time_us * 10**-6
        effective_bandwidth = self.out_gib / time_s

        return {
            "cycles": avg_cycles,
            "time (\u03BCs)": time_us,
            "effective bandwidth (GiB/s)": effective_bandwidth,
        }


def update_bar(bar, metrics):
    r"""Update progress bar with perf metrics"""
    desc = ", ".join(f"{k}: {v:0.1f}" for k, v in metrics.items())
    bar.set_description(desc)


def mpbench(
    operator: str = "gather",
    seed: int = 0,
    synthetic_data: bool = False,
    available_memory_proportion: Optional[float] = None,
    profile_dir: Optional[str] = None,
    cache_dir: str = "mpbench_cache",
    num_repeats: int = 128,
    num_warmup_rounds: int = 4,
    num_sample_rounds: int = 25,
    model_params: Optional[Dict[str, int]] = None,
    use_wandb: bool = False,
):
    """
    Message Passing Benchmark

    :param operator: the operator to benchmark
    :param seed: the random seed to use.
    :param synthetic_data: Use synthetic data on the device to disable I/O.
    :param available_memory_proportion: the AMP budget used for planning ops.
    :param profile_dir: saves the profiling report to the provided location.
    :param cache_dir: saves the executable cache to the provided location
    :param num_repeats: the number of times to invoke the operator on device
    :param num_warmup_rounds: initial set of runs to discard
    :param num_sample_rounds: the number of runs used to average the runtime
    :param num_inputs: the number of inputs
    :param num_features: the number of features
    :param num_outputs: the number of outputs
    :param use_wandb: Use Weights and Biases to log benchmark results.
    """
    torch.manual_seed(seed)
    logger = Logger(locals())
    options = create_options(
        synthetic_data, available_memory_proportion, profile_dir, cache_dir
    )

    if model_params is None:
        # nodes ~ 16, 32, ..., 4096
        num_inputs = [2**e for e in range(4, 13)]

        # embedding size ~ 16, 32, ..., 512
        num_features = [2**e for e in range(4, 10)]

        # edges ~ 32, 64, ..., 32768
        num_outputs = [2**e for e in range(5, 16)]

        grid = product(num_inputs, num_features, num_outputs)
        sweep_params = [ModelParams(*args) for args in grid]
    else:
        sweep_params = [ModelParams(**model_params)]

    for params in sweep_params:
        pop_model = create_model(
            operator=operator,
            num_repeats=num_repeats,
            model_params=params,
            options=options,
        )

        if pop_model is None:
            continue

        for _ in trange(num_warmup_rounds):
            _ = pop_model()

        metrics = PerfMetrics(pop_model, num_repeats)

        for _ in (bar := trange(num_sample_rounds)):
            _ = pop_model()
            values = metrics.update(pop_model.cycleCount())
            logger.log(values)
            update_bar(bar, values)

        logger.commit(asdict(params))


def main():
    CLI(mpbench)


if __name__ == "__main__":
    main()
