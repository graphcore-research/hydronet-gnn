# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from copy import deepcopy

import poptorch
import torch
from torch.testing import assert_close


class TrainingStepper:
    """
    Test utility for comparing training runs between IPU and CPU.
    Usage:
        model = ...
        batch = ...
        model.train()
        stepper = TrainingSteper(model)
        stepper.run(10, batch)
    """

    def __init__(
        self,
        model,
        lr=0.001,
        optimizer=poptorch.optim.Adam,
        options=None,
        rtol=None,
        atol=None,
    ):
        super().__init__()
        model.train()
        self.lr = lr
        self.rtol = rtol
        self.atol = atol
        self.options = poptorch.Options() if options is None else options
        self.setup_cpu(model, optimizer)
        self.setup_ipu(model, optimizer)
        self.check_parameters()

    def setup_cpu(self, model, optimizer):
        self.cpu_model = deepcopy(model)
        self.optimizer = optimizer(self.cpu_model.parameters(), lr=self.lr)

    def setup_ipu(self, model, optimizer):
        self.ipu_model = deepcopy(model)
        ipu_optimizer = optimizer(self.ipu_model.parameters(), lr=self.lr)
        options = self.options
        options.Precision.enableFloatingPointExceptions(True)
        self.training_model = poptorch.trainingModel(
            self.ipu_model, optimizer=ipu_optimizer, options=options
        )

    def assert_close(self, actual, expected, id):
        assert_close(
            actual=actual,
            expected=expected,
            msg=lambda s: f"{id} was not equal\n\n{s}",
            atol=self.atol,
            rtol=self.rtol,
        )

    def check_parameters(self):
        for cpu, ipu in zip(
            self.cpu_model.named_parameters(), self.ipu_model.named_parameters()
        ):
            name, cpu = cpu
            ipu = ipu[1]
            self.assert_close(ipu, cpu, name)

    def cpu_step(self, batch):
        self.optimizer.zero_grad()
        out, loss = self.cpu_model(*batch)
        loss.backward()
        self.optimizer.step()
        return out, loss

    def ipu_step(self, batch):
        out, loss = self.training_model(*batch)
        self.training_model.copyWeightsToHost()
        return out, loss

    def run(self, num_steps, batch):
        cpu_loss = torch.empty(num_steps)
        ipu_loss = torch.empty(num_steps)

        for i in range(num_steps):
            cpu_out, cpu_loss[i] = self.cpu_step(batch)
            ipu_out, ipu_loss[i] = self.ipu_step(batch)
            self.assert_close(ipu_out, cpu_out, "Output")
            self.check_parameters()

        self.assert_close(ipu_loss, cpu_loss, "loss")
