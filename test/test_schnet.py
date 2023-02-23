# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import poptorch
import pytest
import torch
import torch_geometric.nn.models.schnet as pygschnet
from stepper import TrainingStepper
from torch.nn.functional import mse_loss
from torch.testing import assert_close
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import Compose

from hydronet.model import SchNet
from hydronet.transforms import KeySlice

CUTOFF = 10.0
MAX_NUM_NODES = 30

assert_equal = partial(assert_close, rtol=0.0, atol=0.0)


def padding_graph(num_nodes):
    """
    Create a molecule of non-interacting atoms defined as having atomic charge
    of zero to use for padding a mini-batch up to a known maximum size
    """
    assert num_nodes > 0
    return Data(
        z=torch.zeros(num_nodes, dtype=torch.long),
        pos=torch.zeros(num_nodes, 3),
        y=torch.zeros(1),
        num_nodes=num_nodes,
    )


def create_transform():
    def select_target(data):
        # The HOMO-LUMO gap is target 4 in QM9 dataset labels vector y
        target = 4
        data.y = data.y[0, target].flatten()
        return data

    return Compose([select_target, KeySlice()])


@pytest.fixture(params=[2, 8])
def batch(pyg_qm9, request):
    batch_size = request.param
    pyg_qm9.transform = create_transform()
    data_list = list(pyg_qm9[0 : batch_size - 1])
    max_num_nodes = MAX_NUM_NODES * (batch_size - 1)
    num_nodes = sum(d.num_nodes for d in data_list)
    data_list.append(padding_graph(max_num_nodes - num_nodes))
    return Batch.from_data_list(data_list)


class InferenceHarness:
    def __init__(
        self,
        batch_size,
        num_features=32,
        num_gaussians=25,
        num_interactions=2,
    ):
        super().__init__()
        self.seed = 0
        self.batch_size = batch_size
        self.num_features = num_features
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.create_model()
        self.create_reference_model()

    def create_model(self):
        # Set seed before creating the model to ensure all parameters are
        # initialized to the same values as the PyG reference implementation.
        torch.manual_seed(self.seed)
        self.model = SchNet(
            num_features=self.num_features,
            num_gaussians=self.num_gaussians,
            num_interactions=self.num_interactions,
            cutoff=CUTOFF,
            batch_size=self.batch_size,
        )
        self.model.eval()

    def create_reference_model(self):
        # Use PyG implementation as a reference implementation
        torch.manual_seed(self.seed)
        self.ref_model = pygschnet.SchNet(
            hidden_channels=self.num_features,
            num_filters=self.num_features,
            num_gaussians=self.num_gaussians,
            num_interactions=self.num_interactions,
            cutoff=CUTOFF,
        )

        self.ref_model.eval()

    def compare(self, actual, batch):
        expected = self.ref_model(batch.z, batch.pos, batch.batch).view(-1)
        assert_close(actual, expected)

    def test_cpu_padded(self, batch):
        # Run padded model on CPU and check the output agrees with the
        # reference implementation
        actual = self.model(batch.z, batch.pos, batch.ptr[1:-1])
        self.compare(actual, batch)

    def test_ipu(self, batch):
        pop_model = poptorch.inferenceModel(self.model)
        actual = pop_model(batch.z, batch.pos, batch.ptr[1:-1])
        self.compare(actual, batch)


def test_inference(batch):
    batch_size = torch.max(batch.batch).item() + 1
    harness = InferenceHarness(batch_size)
    harness.test_ipu(batch)


def test_inference_cpu(batch):
    batch_size = torch.max(batch.batch).item() + 1
    harness = InferenceHarness(batch_size)
    harness.test_cpu_padded(batch)


def test_training(batch):
    torch.manual_seed(0)
    batch_size = int(batch.batch.max()) + 1
    model = SchNet(
        num_features=16, num_gaussians=15, num_interactions=2, batch_size=batch_size
    )
    model.train()
    stepper = TrainingStepper(model, rtol=1e-4, atol=1e-4)

    batch = (batch.z, batch.pos, batch.ptr[1:-1], batch.y)
    stepper.run(num_steps=3, batch=batch)


def test_loss():
    # Check that loss agrees with mse_loss implementation
    torch.manual_seed(0)
    input = torch.randn(10)
    target = torch.randn(10)
    actual = SchNet.loss(input, target)
    expected = mse_loss(input, target)
    assert_equal(actual, expected)

    # Insert random "padding" zeros.
    mask = torch.randn_like(input) > 0.6
    input[mask] = 0.0
    target[mask] = 0.0
    actual = SchNet.loss(input, target)
    expected = mse_loss(input[~mask], target[~mask])
    assert_equal(actual, expected)
