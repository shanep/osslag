# Releasing osslag

This document describes how to release a new version of osslag.

## Versioning

This project uses [setuptools-scm](https://github.com/pypa/setuptools-scm) for automatic versioning based on git tags. The version is derived directly from git tags at build time.

### Version Format

- **Release versions**: `1.0.0`, `1.1.0`, `2.0.0` (from tags like `v1.0.0`)
- **Development versions**: `1.0.1.dev6+gde04e13` (6 commits after v1.0.0, at commit de04e13)
- **Dirty versions**: `1.0.1.dev6+gde04e13.d20260204` (uncommitted changes present)

## Release Process

### 1. Prepare the Release

Ensure all changes are committed and tests pass:

```bash
# Run tests
uv run pytest

# Run linting
uv run ruff check src/ tests/

# Run type checking
uv run pyright
```

### 2. Create the Release

Choose the appropriate version number following [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0 → 2.0.0): Breaking API changes
- **MINOR** (1.0.0 → 1.1.0): New features, backwards compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, backwards compatible

Create and push the tag:

```bash
# Create an annotated tag
git tag -a v1.1.0 -m "Release v1.1.0"

# Push the tag
git push origin v1.1.0
```

### 3. Create GitHub Release

1. Go to the [GitHub Releases page](https://github.com/shanep/osslag/releases)
2. Click "Draft a new release"
3. Select the tag you just created (e.g., `v1.1.0`)
4. Set the release title (e.g., "v1.1.0")
5. Add release notes describing changes
6. Click "Publish release"

## Verifying the Release

After releasing, verify the version is correct:

```bash
# Install the released version
pip install osslag==1.1.0

# Check the version
python -c "from osslag import get_version; print(get_version())"
```

## Troubleshooting

### Version shows `.dev` suffix after tagging

Make sure you're on the tagged commit:

```bash
git checkout v1.1.0
uv pip install -e .
```

### Version not updating

Reinstall the package to pick up the new version:

```bash
uv pip install -e . --force-reinstall
```

### Checking what version will be generated

```bash
# See what version setuptools-scm will generate
python -m setuptools_scm
```
