# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os.path as osp

import pytest
import torch
from torch_geometric.datasets import QM9
from tqdm import tqdm


@pytest.fixture
def pyg_qm9():
    testdir = osp.abspath(osp.dirname(__file__))
    qm9root = osp.join(testdir, "..", "data", "qm9")
    return QM9(root=qm9root)


@pytest.fixture
def molecule(pyg_qm9):
    # The index of the largest molecule in the QM9 dataset, which looks like:
    # Data(edge_attr=[56, 4], edge_index=[2, 56], idx=[1], name="gdb_57518",
    #      pos=[29, 3], x=[29, 11], y=[1, 19], z=[29])
    max_index = 55967
    return pyg_qm9[max_index]


@pytest.fixture
def qm9_num_nodes(pyg_qm9):
    """
    Fixture for calculating the number of nodes per graph for the QM9 dataset
    Uses pytest caching to store result to speed up testing. Cleared with:
        % pytest --cache-clear
    """
    num_nodes_file = osp.join(pyg_qm9.root, "processed", "num_nodes.pt")

    if osp.exists(num_nodes_file):
        return torch.load(num_nodes_file)

    bar = tqdm(pyg_qm9, desc="Calculating number of nodes")
    num_nodes = [d.num_nodes for d in bar]
    num_nodes = torch.tensor(num_nodes)
    torch.save(num_nodes, num_nodes_file)
    return num_nodes


def pytest_addoption(parser):
    parser.addoption(
        "--no-ipu-hw",
        action="store_true",
        default=False,
        help="Only run tests that do not require IPU hardware",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_ipu: Requires IPU Hardware")


def pytest_runtest_setup(item):
    from poptorch import ipuHardwareIsAvailable

    if any(item.iter_markers("requires_ipu")) and not ipuHardwareIsAvailable():
        pytest.skip("Requires IPU Hardware")
