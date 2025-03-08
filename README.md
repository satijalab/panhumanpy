# package-name
...

## Developer's Guide

### VSCode

This project uses the [Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/containers) for [VSCode](https://code.visualstudio.com/docs) to create reproducible development environments, please ensure that you have the IDE and extension installed. When you open the project in VSCode, the `.devcontainer` folder should be automatically detected and you will prompted you to build the container and
re-open the project inside it. If not, open the Command Palette (`Ctrl+Shift+P` on Windows/Linux or `Cmd+Shift+P` on macOS) and select `Remote-Containers: Reopen in Container`.

### pixi

This project uses [pixi](https://pixi.sh/v0.41.4/) for dependency management and workflow automation. `pixi` uses the contents of the `pyproject.toml` file to generate a cross-platform `pixi.lock` file, providing deterministic, reproducible installations, while seamlessly resolving dependencies from multiple sources (`PyPI`, `conda-forge` etc.).

### pre-commit, isort, & black

This project uses [pre-commit](https://pre-commit.com/) hooks to automatically enforce certain coding standards. Before every commit [isort](https://pycqa.github.io/isort/) and [black](https://black.readthedocs.io/en/stable/index.html) are executed on staged files to ensure that all committed code is formatted consistently. The command can also be invoked manually: 

```bash
pixi run pre-commit
```

Similarly, both tools can be run individually:

```bash
pixi run isort
pixi run black
```

When invoked in this way all files under the `src/` and `tests/` directories will be reformatted, not just staged changes.

### pytest

This project uses [pytest](https://docs.pytest.org/en/stable/) for automated testing. To execute the test suite, run:

```bash
pixi run pytest
```

### flake8, & pyright

This project uses [flake8](https://flake8.pycqa.org/en/latest/) for linting and style enforcement (in addition to `black`). It also uses [pyright](https://microsoft.github.io/pyright/#/) for static type checking. To perform pre-merge checks combining these tools, run:

```bash
pixi run pre-merge
```

Alternatively, you can run them individually:

```bash
pixi run flake8
pixi run pyright
```

### Interactive Development

For interactive development you can spin up a [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) session:

```bash
pixi run jupyter
```

Please take advantage of [jupytext](https://jupytext.readthedocs.io/en/latest/) to write notebooks as plain text
`.py` files and avoid committing `.ipynb` files directly.
