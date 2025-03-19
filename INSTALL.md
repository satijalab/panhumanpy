# Installation Guide

## Prerequisites

- Python 3.12.8
- pixi package manager (optional, recommended)
- pip (Python package installer)

## Installation Methods

### 1. Using pixi (Recommended)

This method ensures you have an appropriate environment:

```bash
# Install pixi if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash
# or with conda: conda install -c conda-forge pixi

# Clone the repository
git clone https://github.com/yourusername/panhumanpy.git
cd panhumanpy

# Install dependencies using pixi
pixi install
```

#### TensorFlow Installation with pixi

After installing the dependencies, you need to install TensorFlow. You have three options:

```bash
# Option 1: Auto-detect GPU/CPU and install appropriate version
pixi run tf-auto

# Option 2: Force CPU-only TensorFlow installation
pixi run tf-cpu

# Option 3: Force GPU TensorFlow installation
pixi run tf-gpu
```

#### Using the Package with pixi

After installation, you need to activate the pixi environment each time you want to use the package:

```bash
cd panhumanpy  # Navigate to your package directory
pixi shell     # Activate the environment

# Now you can run your Python scripts
python your_script.py

# Or launch an interactive Python session
python

# Or use Jupyter Notebook (if installed)
jupyter notebook
```

### 2. Using pip

If you prefer pip, you can install the package and its dependencies directly:

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

# Install the package in editable mode with dependencies
pip install -e .

# Note: This will automatically install dependencies specified in pyproject.toml
```

#### TensorFlow Installation with pip

After installing the package, install the appropriate TensorFlow version:

```bash
# For CPU-only systems:
pip install tensorflow-cpu==2.17.0

# For systems with NVIDIA GPU:
pip install tensorflow[and-cuda]==2.17.0
```

## Dependencies

The package requires the following main dependencies:

### Core Dependencies
- Python 3.12.8
- numpy 1.26.4
- pandas 2.2.3
- scikit-learn 1.6.0
- h5py 3.12.1
- anndata 0.10.9
- umap-learn 0.5.7
- numba 0.60.0
- distances
- keras
- tensorflow 2.17.0 (CPU or GPU version)

All dependencies will be automatically installed by pixi, or need to be installed manually with pip as shown above.

## Verifying Installation

After installation, you can verify it worked by importing the package:

```python
import panhumanpy
print(panhumanpy.__version__)
```

Or run the tests:

```bash
# If using pixi:
pixi run test

# If using pip:
pytest tests/
```

## Adding Custom Dependencies

If you need additional packages for your own work:

```bash
# While in the activated pixi environment
pixi shell
pip install additional-package

# Or with pip in your normal environment
pip install additional-package
```

## Common Issues

### TensorFlow GPU Issues

If you encounter issues with the GPU version of TensorFlow:

1. Make sure you have compatible NVIDIA drivers installed
2. Check that your CUDA version is compatible with TensorFlow 2.17
3. Check your GPU driver version using 
    ```bash
    nvidia-smi
    ```
4. Ensure that your system is correctly configured for GPU usage. In particular, if you have multiple GPUs or need to restrict TensorFlow to a specific device, you may need to set the CUDA_VISIBLE_DEVICES environment variable. For example:
    ```bash
    #On Unix/Linux/macOS, run:
    export CUDA_VISIBLE_DEVICES=0

    #On Windows (Command Prompt), run:
    set CUDA_VISIBLE_DEVICES=0
    ```
    This variable controls which GPU(s) TensorFlow will see. If you do not set this variable, TensorFlow might use a GPU by default or may run into issues if multiple GPUs are available.

5. Try the CPU-only version instead

### Version Conflicts

If you encounter dependency conflicts with pip, consider using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Then install the package with dependencies
pip install -e /path/to/panhumanpy

# Or if you have a wheel file:
pip install panhumanpy-0.1.0-py3-none-any.whl

# Then install the appropriate TensorFlow version
pip install tensorflow-cpu==2.17.0  # or GPU version
```

## Getting Help

If you encounter any issues during installation:
1. Check the [GitHub Issues](https://github.com/rsatija/panhumanpy/issues)
2. Create a new issue with details about your system and the error message
3. Contact the maintainers