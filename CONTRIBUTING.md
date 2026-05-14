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

### Versioning Protocol

This project follows a generalised semantic versioning where major versions are associated with a constellation as a codename. 

**Format:** `MAJOR.MINOR.PATCH (Constellation Name)`

#### Version Types:
- **MAJOR (X.0.0):** New default model added, and/or breaking changes to the API → New constellation name
- **MINOR (X.Y.0):** New features/functionality 
- **PATCH (X.Y.Z):** Bug fixes, small improvements

#### Current Version:
- 0.4.0 (Andromeda)

#### Version Philosophy:
- **New default models, and/or breaking changes to API = Major versions** because they represent ("soft") breaking changes.
- **Version bumps are at maintainer discretion** - not every PR requires a version change.
- When versions are updated, `pyproject.toml`, `src/panhumanpy/__init__.py`, `README.md`, and `CONTRIBUTING.md` must be kept in sync.
- Use versioning script for version bumps executed in a consistent fashion.

#### Rules:
1. The default model version number must match the package major version
   - Package `0.2.1` → Model `v0` as default
   - Package `1.3.2` → Model `v1` as default
2. Model artifacts are stored in `src/panhumanpy/_tools/v{i}/`
3. When bumping major package version, create new `v{i}` directory with updated models and set default model version in ANNotate.py appropriately.

#### Example:
```python
# In __init__.py
__version__ = "0.2.1"

# In ANNotate.py  
model_version_default = 'v0'  # Must match major version (0)
```


### Using the Version Bump Script

For consistent version updates, use the provided script:

```bash
# Check current version
python scripts/bump_version.py current

# Check version consistency across files
python scripts/bump_version.py check

# Bug fixes (patch)
python scripts/bump_version.py patch

# New features (minor)
python scripts/bump_version.py minor

# New models or breaking changes (major)
python scripts/bump_version.py major Cassiopeia

# Run tests that will ensure version consistency across scripts and docs
pytest

# Push changes and tags on git
git diff
git add . && git commit -m 'Bump version to 0.2.0'
git tag v0.2.0-andromeda
git push --follow-tags # if on branch main
git push origin version_bump_branch --tags # if on a separate branch
```