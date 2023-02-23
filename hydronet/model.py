# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn.models.schnet as pyg_schnet
from poptorch import (
    OverlapMode,
    identity_loss,
    set_overlap_for_input,
    set_overlap_for_output,
)
from torch import Tensor
from torch_geometric.nn import to_fixed_size


class SchNet(torch.nn.Module):
    def __init__(
        self,
        num_features: int = 100,
        num_interactions: int = 4,
        num_gaussians: int = 25,
        k: int = 28,
        cutoff: float = 6.0,
        batch_size: Optional[int] = None,
        overlap_io: bool = False,
        **kwargs
    ):
        """
        :param num_features (int): The number of hidden features used by both
            the atomic embedding and the convolutional filters (default: 100).
        :param num_interactions (int): The number of interaction blocks
            (default: 4).
        :param num_gaussians (int): The number of gaussians used in the radial
            basis expansion (default: 25).
        :param k (int): The number of nearest neighbors used in calculating the
            pairwise interaction graph (default: 28).
        :param cutoff (float): Cutoff distance for interatomic interactions
            which must match the one used to build the radius graphs
            (default: 6.0).
        :param batch_size (int, optional): The number of molecules in the
            batch. This can be inferred from the batch input when not supplied.
        """
        super().__init__()
        self.k = k
        self.model = pyg_schnet.SchNet(
            hidden_channels=num_features,
            num_filters=num_features,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            interaction_graph=KNNInteractionGraph(k, cutoff),
            **kwargs
        )

        self.overlap_io = overlap_io
        if batch_size is not None:
            self.model = to_fixed_size(self.model, batch_size)

        self.hparams = {
            "num_features": num_features,
            "num_interactions": num_interactions,
            "num_gaussians": num_gaussians,
            "cutoff": cutoff,
            "batch_size": batch_size,
            "k": k,
        }

    def reset_parameters(self):
        self.model.reset_parameters()

    def hyperparameters(self):
        return self.hparams

    def forward(self, z, pos, ptr=None, target=None):
        """
        Forward pass of the SchNet model

        :param z: Tensor containing the atomic numbers for each atom in the
            batch. Vector of integers with size [num_atoms].
        :param pos: Tensor containing the atomic coordinates.
        :param edge_index: Tensor containing the indices defining the
            interacting pairs of atoms in the batch. Matrix of integers with
            size [2, num_edges]
        :param ptr: Tensor assigning each atom within a batch to a molecule.
            This is used to perform per-molecule aggregation to calculate the
            predicted energy. Vector of integers with size [num_atoms]
        :param target (optional): Tensor containing the target to
            use for evaluating the mean-squared-error loss when training.
        """
        if self.overlap_io:
            z = overlap(z)
            pos = overlap(pos)
            ptr = overlap(ptr)
            target = overlap(target)

        z = z.int()
        batch = ptr_to_batch(ptr, z.numel())
        args = [collapse(t) for t in (z, pos, batch, target)]

        args, target = args[:-1], args[-1]
        out = self.model(*args).view(-1)

        if not self.training:
            return out

        loss = self.loss(out, target)

        if self.overlap_io:
            out = overlap(out, isinput=False)
            loss = overlap(loss, isinput=False)

        return out, loss

    @staticmethod
    def loss(input, target):
        """
        Calculates the mean squared error

        This loss assumes that zeros are used as padding on the target so that
        the count can be derived from the number of non-zero elements.
        """
        loss = F.mse_loss(input, target, reduction="sum")
        N = (target != 0.0).to(loss.dtype).sum()
        loss = loss / N
        return identity_loss(loss, reduction="none")

    @staticmethod
    def create_with_fss(**kwargs):
        """
        Creates the SchNet model with the faster implementation of the shifted
        softplus activation.
        """
        model = SchNet(**kwargs)
        FastShiftedSoftplus.replace_activation(model)
        return model


def ptr_to_batch(ptr, num_nodes):
    if ptr is None:
        return ptr

    ptr = collapse(ptr.long())
    batch = ptr.new_zeros(num_nodes)
    batch = batch.index_fill(dim=0, index=ptr, value=1)
    batch[0] = 0
    return torch.cumsum(batch, dim=0)


class FastShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        u = torch.log1p(torch.exp(-x.abs()))
        v = torch.clamp_min(x, 0.0)
        return u + v - self.shift

    @staticmethod
    def replace_activation(module: torch.nn.Module):
        for name, child in module.named_children():
            if isinstance(child, pyg_schnet.ShiftedSoftplus):
                setattr(module, name, FastShiftedSoftplus())
            else:
                FastShiftedSoftplus.replace_activation(child)


def collapse(input: Tensor) -> Tensor:
    # Collapse any leading batching dimensions
    return input if input is None else input.squeeze(0)


def overlap(input: Tensor, isinput: bool = True) -> Tensor:
    overlapfunc = set_overlap_for_input if isinput else set_overlap_for_output

    return (
        input
        if input is None
        else overlapfunc(input, OverlapMode.OverlapDeviceIterationLoop)
    )


class KNNInteractionGraph(torch.nn.Module):
    def __init__(self, k: int, cutoff: float = 10.0):
        super().__init__()
        self.k = k
        self.cutoff = cutoff

    def forward(self, pos: Tensor, batch: Tensor):
        """
        k-nearest neighbors without dynamic tensor shapes

        :param pos (Tensor): Coordinates of each atom with shape
            [num_atoms, 3].
        :param batch (LongTensor): Batch indices assigning each atom to
                a separate molecule with shape [num_atoms]

        This method calculates the full num_atoms x num_atoms pairwise distance
        matrix. Masking is used to remove:
            * self-interaction (the diagonal elements)
            * cross-terms (atoms interacting with atoms in different molecules)
            * atoms that are beyond the cutoff distance

        Finally topk is used to find the k-nearest neighbors and construct the
        edge_index and edge_weight.
        """
        pdist = F.pairwise_distance(pos.unsqueeze(1), pos, eps=0)
        rows = arange_like(batch.shape[0], batch).view(-1, 1)
        cols = rows.view(1, -1)
        diag = rows == cols
        cross = batch.view(-1, 1) != batch.view(1, -1)
        outer = pdist > self.cutoff
        mask = diag | cross | outer
        pdist = pdist.masked_fill(mask, self.cutoff)
        edge_weight, indices = torch.topk(-pdist, k=self.k)
        rows = rows.expand_as(indices)
        edge_index = torch.vstack([indices.flatten(), rows.flatten()])
        return edge_index, -edge_weight.flatten()


def arange_like(n: int, ref: Tensor) -> Tensor:
    return torch.arange(n, device=ref.device, dtype=ref.dtype)
