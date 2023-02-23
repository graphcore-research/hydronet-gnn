# Contributing to hydronet-gnn


## Setting up a development environment
We recommend using the conda package manager as this can automatically enable
the Poplar SDK.  This is particularly useful in VS Code which can automatically
activate the conda environment in a variety of scenarios:
* visual debugging
* running quick experiments in an interactive Jupyter window
* using VS code for Jupyter notebook development.

The following assumes that you have already setup an install of conda and that
the conda command is available on your system path.  Refer to your preferred conda
installer:
* [miniforge installation](https://github.com/conda-forge/miniforge#install)
* [conda installation documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Create a new conda environment with the same python version as you OS.
   For example, on ubuntu 20 use `python=3.8.10`
   ```bash
   conda create -n hydronet python=3.8.10
   ```

2. Activate the environment and store a persistent environment variable for the
   location of the downloaded Poplar SDK. This assumes that
   you have already downloaded the Poplar SDK.  The following example uses an
   environment variable `$POPLAR_SDK` to store the root folder for the SDK.
   ```bash
   conda activate hydronet
   conda env config vars set POPLAR_SDK=/path/to/poplar/sdk
   ```

3. You have to reactivate the conda environment to use the `$POPLAR_SDK`
   variable the environment. So we do this and then use it to install the
   poptorch wheel.
   ```bash
   conda deactivate
   conda activate hydronet
   pip install $POPLAR_SDK/poptorch*.whl
   ```
   This will install the poptorch wheel along with the compatible version of
   PyTorch.

4. Setup the conda environment to automatically enable the Poplar SDK whenever
   the environment is activated.
   ```bash
   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
   echo "source $POPLAR_SDK/enable" > $CONDA_PREFIX/etc/conda/activate.d/enable.sh
   ```

5. Check that everything is working by printing the poptorch version string.
   ```bash
   python3 -c "import poptorch;print(poptorch.__version__)"
   ```

5. Install requirements and use a local editable install of the project.
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```


6. Install the pre-commit hooks
   ```bash
   pre-commit install
   ```

7. Create a feature branch, make changes, and when you commit them the
   pre-commit hooks will run.
   ```bash
   git checkout -b feature
   ...
   git push --set-upstream origin feature
   ```
   The last command will prints a link that you can follow to open a PR.


## Testing
Run all the tests using `pytest`
```bash
pytest
```

## Benchmarks

We use both:
* [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/index.html)
  to register micro-benchmarks that also run as part of the normal `pytest` run.
  We recommend that any benchmarks run in under one minute so that the full
  unittest suite can be executed in fast develop-debug-update loops.

* [py-spy](https://github.com/benfred/py-spy) is a sampling profiling tool that
  is automatically installed in the docker development environment. This is
  useful for generating flame graphs to identify performance bottlenecks.

Possible workflow for performance tuning could look like:

1. Write a micro-benchmark and run it to collect the execution statistics:
```bash
pytest -s <test to run> --benchmark-compare --benchmark-autosave
```
The additional arguments will save an execution report in the `.benchmarks`
folder that will be our baseline execution.

2. Use `py-spy` to collect a flamegraph of the benchmark:
```bash
py-spy record -o profile.svg -- python -m pytest <test to run>
```

3. Scrutinise the flamegraph in a browser.

4. Make a change to the codebase based on the flamegraph and repeat the above.

5. The above can go on indefinitely so it would be worth determining some
   critical performance criteria that needs to be achieved before moving on.
