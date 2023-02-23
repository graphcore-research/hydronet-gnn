# Fast and Accurate Predictions from 3D Molecular Structures


Molecular property calculations are the bedrock of chemical physics.
High-fidelity *ab initio* modeling techniques for computing the molecular
properties can be prohibitively expensive, and motivate the development of
machine-learning models that make the same predictions more efficiently.
Training graph neural networks over large molecular databases introduces unique
computational challenges such as the need to process millions of small graphs
with variable size and support communication patterns that are distinct from
learning over large graphs such as social networks.

## Installation

1. Prepare the PopTorch environment. Install the Poplar SDK following the
   Getting Started guide for your IPU system. Make sure to source the
   `enable.sh` scripts for Poplar and PopART and activate a Python virtualenv
   with the PopTorch wheel installed.
2. Install additional Python packages specified in requirements.txt
```:bash
pip3 install -r requirements.txt
```
3. Install this project from source
```:bash
pip3 install -e .
```
The `-e` option can be omitted and allows local edits to the project. Refer to the
[contribution guide](./CONTRIBUTING.md) for instructions to setup a development
environment.

## Running
`hydronet-bench` is the main script for running a comprehensive performance experiment
on the IPU. The full usage is replicated below for reference.  This script
accepts a `--config` argument to specify a `yml` file containing the options to
use for the experiment. For example, to run the benchmark with graph packing:
```:bash
hydronet-bench --config configs/qm9_packed.yml
```

Similarly, to run the benchmark with padding applied to every molecule:
```:bash
hydronet-bench --config configs/qm9_padded.yml
```

Configurations set in a `yml` file can be used in combination with command-line
arguments:
```:bash
hydronet-bench --config configs/qm9_packed.yml --use_wandb true
```
The above command will run the packing benchmark with the logging to
[Weights and Biases](https://docs.wandb.ai/) enabled.

### Complete Usage
Below are the supported command-line arguments for `hydronet-bench`.

```:bash
hydronet-bench --help
usage: hydronet-bench [-h] [--config CONFIG] [--print_config[=flags]]
                      [--dataset {qm9,hydronet,hydronet_medium,hydronet_small}]
                      [--use_packing {true,false}] [--synthetic_data {true,false}]
                      [--available_memory_proportion AVAILABLE_MEMORY_PROPORTION]
                      [--combined_batch CONFIG]
                      [--combined_batch.model_batch_size MODEL_BATCH_SIZE]
                      [--combined_batch.device_iterations DEVICE_ITERATIONS]
                      [--combined_batch.replication_factor REPLICATION_FACTOR]
                      [--combined_batch.gradient_accumulation GRADIENT_ACCUMULATION]
                      [--model_params CONFIG]
                      [--model_params.num_features NUM_FEATURES]
                      [--model_params.num_interactions NUM_INTERACTIONS]
                      [--model_params.num_gaussians NUM_GAUSSIANS]
                      [--model_params.cutoff CUTOFF]
                      [--module {schnet,schnet_fss,interaction}]
                      [--learning_rate LEARNING_RATE] [--num_epochs NUM_EPOCHS]
                      [--profile_dir PROFILE_DIR] [--prefetch_depth PREFETCH_DEPTH]
                      [--num_io_tiles NUM_IO_TILES]
                      [--task {train,profile,dataloading}] [--seed SEED]
                      [--max_workers MAX_WORKERS] [--use_async_loader {true,false}]
                      [--merge_all_reduce {true,false}] [--use_wandb {true,false}]

IPU Benchmark of SchNet model

optional arguments:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and
                        exit. The optional flags customizes the output and are one or
                        more keywords separated by comma. The supported flags are:
                        comments, skip_default, skip_null.
  --dataset {qm9,hydronet,hydronet_medium,hydronet_small}
                        qm9, hydronet, hydronet_medium, or hydronet_small (type:
                        BenchDataset, default: qm9)
  --use_packing {true,false}
                        Apply graph packing to the dataset. (type: bool, default:
                        True)
  --synthetic_data {true,false}
                        Use synthetic data on the device to disable I/O. (type: bool,
                        default: False)
  --available_memory_proportion AVAILABLE_MEMORY_PROPORTION
                        the AMP budget used for planning ops. (type: Union[float,
                        null], default: null)
  --module {schnet,schnet_fss,interaction}
                        module to benchmark, schnet, schnet_fss, or interaction (type:
                        BenchModule, default: schnet_fss)
  --learning_rate LEARNING_RATE
                        The learning rate used by the optimiser. (type: float,
                        default: 0.001)
  --num_epochs NUM_EPOCHS
                        The number of epochs to benchmark. (type: int, default: 25)
  --profile_dir PROFILE_DIR
                        Run a single training step with profiling enabled and saves
                        the profiling report to the provided location. (type:
                        Union[str, null], default: null)
  --prefetch_depth PREFETCH_DEPTH
                        The depth of the buffer used for prefetching batches from the
                        host. (type: int, default: 3)
  --num_io_tiles NUM_IO_TILES
                        the number of tiles used with overlapped IO. (type: Union[int,
                        null], default: 0)
  --task {train,profile,dataloading}
                        the benchmark task to run. (type: Task, default: train)
  --seed SEED           the random seed to use. (type: int, default: 0)
  --max_workers MAX_WORKERS
                        the maximum number of data loader workers to use. (type: int,
                        default: 32)
  --use_async_loader {true,false}
                        Enables poptorch.DataLoaderMode.Async (type: bool, default:
                        True)
  --merge_all_reduce {true,false}
                        Enables PopART merging of all reduce collectives (type: bool,
                        default: True)
  --use_wandb {true,false}
                        Use Weights and Biases to log benchmark results. (type: bool,
                        default: True)

CombinedBatch(model_batch_size: int, device_iterations: int = 1, replication_factor: int = 1, gradient_accumulation: int = 1):
  --combined_batch CONFIG
                        Path to a configuration file.
  --combined_batch.model_batch_size MODEL_BATCH_SIZE
                        (type: int, default: 16)
  --combined_batch.device_iterations DEVICE_ITERATIONS
                        (type: int, default: 1)
  --combined_batch.replication_factor REPLICATION_FACTOR
                        (type: int, default: 16)
  --combined_batch.gradient_accumulation GRADIENT_ACCUMULATION
                        (type: int, default: 8)

ModelParams(num_features: int = 100, num_interactions: int = 4, num_gaussians: int = 25, cutoff: float = 6.0):
  --model_params CONFIG
                        Path to a configuration file.
  --model_params.num_features NUM_FEATURES
                        (type: int, default: 100)
  --model_params.num_interactions NUM_INTERACTIONS
                        (type: int, default: 4)
  --model_params.num_gaussians NUM_GAUSSIANS
                        (type: int, default: 25)
  --model_params.cutoff CUTOFF
                        (type: float, default: 6.0)
```

## Profiling
Graphcore have developed two profiling tools that can help identify bottlenecks:

* [PopVision Graph Analyser](https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/)
* [PopVision System Analyser](https://docs.graphcore.ai/projects/system-analyser-userguide/en/latest/)

For example, the following command will collect a graph analyser report for a
single epoch with dataset packing:
```:bash
hydronet-bench --config configs/qm9_packed.yml --num_epochs 1 --profile_dir graph_profile
```
This technique is used to understand the execution of the model on the
IPU.  The system analyser complements this by providing insight into the
execution of the data pipeline on the host.  Set the following environment
variable to capture a system analyser profile:
```:bash
export PVTI_OPTIONS='{"enable":"true", "directory":"system_profile"}'
```
Running your model with this environment variable set will enable
instrumentation in the Poplar SDK to capture timing information for the process
of orchestrating data movement from the host to the IPU. Additional timing
information can be collected by using the
[poptorch.profiling.Channel](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#poptorch.profiling.Channel)
context
manager.

## IPU Resources
* [Graphcore Documentation](https://docs.graphcore.ai/en/latest/)
* [PopTorch Documentation](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/index.html)
* [PopTorch Examples and Tutorials](https://docs.graphcore.ai/en/latest/examples.html#pytorch)
* [Memory and Performance Optimisation](https://docs.graphcore.ai/projects/memory-performance-optimisation/en/latest/index.html)



## Contributing

[Contribution guidelines for hydronet-gnn](./CONTRIBUTING.md)

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2211.13853,
  doi = {10.48550/ARXIV.2211.13853},

  url = {https://arxiv.org/abs/2211.13853},

  author = {Helal, Hatem and Firoz, Jesun and Bilbrey, Jenna and Krell, Mario Michael and Murray, Tom and Li, Ang and Xantheas, Sotiris and Choudhury, Sutanay},a

  keywords = {Machine Learning (cs.LG), Hardware Architecture (cs.AR), Chemical Physics (physics.chem-ph), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Physical sciences, FOS: Physical sciences},
```
