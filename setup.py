# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from pathlib import Path

from setuptools import setup

__version__ = "0.0.1"


def read_requirements(filename):
    pwd = Path(__file__).parent.resolve()
    txt = (pwd / filename).read_text(encoding="utf-8").split("\n")

    def filter_func(string):
        return len(string) > 0 and not string.startswith(("#", "-"))

    return list(filter(filter_func, txt))


setup(
    name="hydronet-gnn",
    version=__version__,
    description="HydroNet GNN - Molecular Property Prediction from 3D Structure",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    license="MIT License",
    author="Graphcore Research",
    author_email="hatemh@graphcore.ai",
    url="https://github.com/graphcore-research/hydronet-gnn",
    project_urls={
        "Code": "https://github.com/graphcore-research/hydronet-gnn",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.8",
    packages=["hydronet"],
    entry_points={
        "console_scripts": [
            "hydronet-bench = hydronet.bench:main",
            "hydronet-mpbench = hydronet.mpbench:main",
        ]
    },
)
