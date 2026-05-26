# panhumanpy

**Current version: 0.4.0 (Andromeda)**

A package for cell annotation using Azimuth Neural Network.

## Prerequisites

- `python` >=3.9
- `pip`
- `git`

## Installation

To install the base version of the package (with CPU support only), run:

```bash
pip install panhumanpy
```

or to install from GitHub, run:

```bash
pip install git+https://github.com/satijalab/panhumanpy.git
```

If you require GPU acceleration for enhanced performance on compatible hardware, install the package with GPU dependencies:

```bash
pip install panhumanpy[gpu]
```

or from GitHub:


```bash
pip install git+https://github.com/satijalab/panhumanpy.git#egg=panhumanpy[gpu]
```

## Model Versions

panhumanpy uses versioned models corresponding to major package releases. The package defaults to model 'v{i}' where i is the major package version. For example for panhumanpy 0.2.1 (Andromeda), the default model version is 'v0'. For most users, the default version is recommended. The user can also choose to use a different model version as outlined in the tutorial mentioned below. 

Currently available model versions: 'v0', 'v1'

## Model Weights

Model weights are hosted on Zenodo and downloaded automatically on first use, cached in `~/.cache/panhumanpy/`. No manual download is required.

| Field | Detail |
|---|---|
| DOI | https://doi.org/10.5281/zenodo.20401417 |
| Models | v0, v1 |
| License | CC BY 4.0 |

## Cell Ontology Mapping

panhumanpy includes a built-in crosswalk that maps Pan-human Azimuth cell type annotations to [Cell Ontology](https://obofoundry.org/ontology/cl.html) (CL) terms. This mapping is versioned alongside the model and can be applied to annotation outputs via the `map_to_cell_ontology` function in `ANNotate_tools` or the `map_to_cell_ontology` method on `AzimuthNN` and `AzimuthNN_base`.

**Crosswalk provenance:**

| Field | Detail |
|---|---|
| Title | Crosswalk of Pan-human Azimuth Types annotated cells to Cell Ontology |
| Author | Aleix Puig-Barbe |
| Author ORCID | 0000-0001-6677-8489 |
| Reviewers | Bruce Herr II, Katy Borner, Jie Zheng |
| Reviewer ORCIDs | 0000-0002-6703-7647, 0000-0002-3321-6137, 0000-0002-2999-0103 |
| Data DOI | https://doi.org/10.48539/HBM727.TLKL.237 |
| Date | December 15, 2025 |
| Version | v1.1 |

## Tutorial

For an introductory tutorial, please check out this [notebook](https://github.com/satijalab/panhumanpy/blob/main/tutorial_panhumanpy.ipynb).
