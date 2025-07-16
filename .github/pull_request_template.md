## Description
Checklist of guidelines to go through when making a PR.

## Type of Change
- [ ] Bug fix (patch version)
- [ ] New or updated feature (minor version)  
- [ ] New model (major version + new constellation for version name)
- [ ] Documentation update
- [ ] Internal refactoring

**Does this PR introduce a breaking change to the *public API* of existing features/models?**
- [ ] Yes (If yes, this *will* require a Major version bump. Explain in "Additional Notes")
- [ ] No

## Pre-Merge Checklist
**Please check each item before requesting review:**

### Code Quality
- [ ] Tests pass locally: `python -m pytest tests/`
- [ ] New functionality has tests written
- [ ] No debugging code left behind (print statements, breakpoints)
- [ ] Package installs cleanly: `pip install -e .`

### Documentation
- [ ] README.md updated for user-facing changes and model version updates if necessary
- [ ] Docstrings added for new functions
- [ ] Tutorial notebook updated if needed

### Model Changes (if applicable)
- [ ] New models tested with sample data
- [ ] Model files added to appropriate `_tools/` subdirectory
- [ ] Backward compatibility maintained (old models still work)

### Git Hygiene
- [ ] Branch is up to date with main
- [ ] No `__pycache__` files committed
- [ ] Commit messages are descriptive
- [ ] No merge conflicts

## Version Impact (Maintainer Decision)
- [ ] No version change needed
- [ ] Should trigger version bump:
  - [ ] Patch (bug fix)
  - [ ] Minor (new or updated feature)  
  - [ ] Major (breaking change to existing API, **or a new model** which is a "soft" breaking change) - change constellation for version name

## Testing
How was this change tested?

## Additional Notes

