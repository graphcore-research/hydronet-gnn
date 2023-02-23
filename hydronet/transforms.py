# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Compose, Pad

from .data_utils import data_keys


class QM9EnergyTarget(BaseTransform):
    """
    Data transform to extract the energy target from the PyG QM9 dataset.

    The QM9 dataset consists of a total of 19 regression targets. This
    transform updates the regression targets stored in data.y to only include
    the internal energy at 0K (eV) to use as the target for training.

    Expected input:
        data.y is a vector with shape [1, 19]

    Transformed output:
        data.y is as scalar with shape torch.Size([])
    """

    ENERGY_TARGET = 7

    def __init__(self, debug: bool = False):
        self.debug = debug

    def validate(self, data):
        assert (
            hasattr(data, "y")
            and isinstance(data.y, torch.Tensor)
            and data.y.shape == (1, 19)
        ), "Invalid data input. Expected data.y == Tensor with shape [1, 19]"

    def __call__(self, data):
        self.validate(data)
        data.y = data.y[0, self.ENERGY_TARGET]
        return data


class StandardizeEnergy(BaseTransform):
    """
    Data transform for energy standardization

    This transform rescales the molecular total energy to an energy per-atom
    that does not include the single-atom energies. This is done by looking up
    a reference value for the single-atom energy for each atom. These values
    are added up over all atoms in the molecule and subtracted from the
    molecular total energy. Finally this difference is rescaled by the number
    of atoms in the sample. More succinctly:

    E_per_atom = (E_molecule - sum(E_atomref)) / num_atoms

    This transform is motivated by training a SchNet network with a dataset
    that contains molecules with a high variation in the number of atoms. This
    will be directly correlated with a large variation in the total energy
    regression target. Applying this transform is expected to both accelerate
    and stabilize the training process.
    """

    def __init__(self):
        super().__init__()
        self.atomref = self.energy_atomref()

    @staticmethod
    def energy_atomref():
        """
        Single-atom reference energies for atomic elements in QM9 data. See:

            torch_geometric/datasets/qm9.py: https://git.io/JP6iU

        Limited to H, C, N, O, and F elements which are the only atoms present
        in the QM9 dataset.
        """
        refs = [
            -13.61312172,
            -1029.86312267,
            -1485.30251237,
            -2042.61123593,
            -2713.48485589,
        ]

        out = torch.zeros(100)
        out[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor(refs)
        return out

    def _sum_atomrefs(self, z):
        refs = self.atomref[z]
        mask = refs == 0.0
        assert ~torch.any(mask), f"Invalid element type(s): {z[mask].tolist()}"
        return refs.sum()

    def __call__(self, data):
        num_atoms = data.z.numel()
        sum_atomrefs = self._sum_atomrefs(data.z)
        data.y = (data.y - sum_atomrefs) / num_atoms
        return data

    def inverse(self, z, y):
        """
        Performs the inverse of the standardize energy transform.

        :param z (Tensor [num_atoms]): The atomic numbers for the molecule
        :param y (Tensor): The standardized energy to invert.
        """
        return y * z.numel() + self._sum_atomrefs(z)


class KeySlice(BaseTransform):
    def __init__(self, include_keys=data_keys()) -> None:
        super().__init__()
        if "num_nodes" not in include_keys:
            # Include num_nodes to prevent warnings with async loader
            include_keys = include_keys + ("num_nodes",)

        self.include_keys = include_keys

    def __call__(self, data):
        values = [getattr(data, k) for k in self.include_keys]
        kwargs = dict([*zip(self.include_keys, values)])
        return Data(**kwargs)


def create_transform(
    use_qm9_energy: bool = False,
    use_standardized_energy: bool = False,
    use_padding: bool = False,
    max_num_nodes: Optional[int] = None,
):
    """
    Creates a sequence of transforms defining a data pre-processing pipeline

    :param use_qm9_energy (bool): Use the QM9EnergyTarget transform
        (default: False).
    :param use_standardized_energy (bool): Use the StandardizeEnergy transform
        (default: False).

    :returns: A composite transform
    """
    transforms = []
    if use_qm9_energy:
        transforms.append(QM9EnergyTarget())

    if use_standardized_energy:
        transforms.append(StandardizeEnergy())

    transforms.append(KeySlice())

    if use_padding:
        transforms.append(Pad(max_num_nodes=max_num_nodes, exclude_keys="y"))

    return Compose(transforms)
