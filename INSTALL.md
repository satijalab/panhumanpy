# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Optional: pixi package manager

## Installation Methods

### 1. Using pixi (Recommended)

This method ensures you have the exact same environment as development:

```bash
# Install pixi if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://github.com/yourusername/panhumanpy.git
cd panhumanpy

# Install using pixi
pixi install
```

### 2. Using pip

#### From PyPI (not yet available):
```bash
pip install panhumanpy
```

#### From wheel file:
```bash
# Download the wheel file and run:
pip install panhumanpy-0.1.0-py3-none-any.whl
```

#### From source:
```bash
# Clone the repository
git clone https://github.com/yourusername/panhumanpy.git
cd panhumanpy

# Install in editable mode
pip install -e .
```

## Dependencies

The package requires the following main dependencies:
- numpy >= 2.0
- matplotlib >= 3.9
- pandas >= 2.2
- torch >= 2.5.0
- tensorboard >= 2.18
- anndata >= 0.10.9
- scikit-learn
- umap-learn

All dependencies will be automatically installed by pip or pixi.

## Verifying Installation

After installation, you can verify it worked by running Python and importing the package:

```python
import panhumanpy
```

Or run the tests:

```bash
# If using pixi:
pixi run test

# If using pip:
pytest tests/
```

## Common Issues

### GPU Support
By default, the package installs with CPU-only PyTorch. If you need GPU support:

```bash
# First uninstall existing torch
pip uninstall torch

# Then install torch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118  # for CUDA 11.8
```

### Version Conflicts
If you encounter dependency conflicts, we recommend using pixi or a virtual environment:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Then install the package
pip install panhumanpy-0.1.0-py3-none-any.whl
```

## Getting Help

If you encounter any issues during installation:
1. Check the [GitHub Issues](https://github.com/yourusername/panhumanpy/issues)
2. Create a new issue with details about your system and the error message
3. Contact the maintainers 