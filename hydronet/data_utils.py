# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import os


def dataroot(folder: str):
    """
    Root directory where downloaded and pre-processed datasets are stored

    Defaults to common root data subfolder within the schnet project.

    >>> dataroot("qm9")
        '/.../data/qm9'

    :param folder (str): Name of the must be non-empty.
    """
    assert (
        isinstance(folder, str) and len(folder) > 0
    ), f"Invalid folder: '{folder}'. Argument must be a non-empty string."

    this_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(this_dir, "data", folder)


def data_keys():
    """
    Data keys used by default by the SchNet network

        * z Tensor[num_atoms]: atomic number
        * pos Tensor[num_atoms, 3]: the atomic coordinates
        * edge_index Tensor[2, num_edges]: the adjacency list defining the
             edges.
        * y Tensor[]: The energy target as a scalar.
        * num_nodes int: the number of atoms in the graph
    """
    return ("z", "pos", "y")
