# panhumanpy

**Current version: 0.2.1 (Andromeda)**

A package for cell annotation using Azimuth Neural Network.

## Prerequisites

- `python` >=3.9
- `pip`
- `git`

## Installation

To install the base version of the package (with CPU support only), run:

```bash
pip install git+https://github.com/satijalab/panhumanpy.git
```

If you require GPU acceleration for enhanced performance on compatible hardware, install the package with GPU dependencies:

```bash
pip install git+https://github.com/satijalab/panhumanpy.git#egg=panhumanpy[gpu]
```

## Model Versions

panhumanpy uses versioned models corresponding to major package releases. The package defaults to model 'v{i}' where i is the major package version. For example for panhumanpy 0.2.1 (Andromeda), the default model version is 'v0'. For most users, the default version is recommended. The user can also choose to use a different model version as outlined in the tutorial mentioned below. 

Currently available model versions: 'v0', 'v1'

## Tutorial

For an introductory tutorial, please check out this [notebook](https://github.com/satijalab/panhumanpy/blob/main/tutorial_panhumanpy.ipynb).
