# Contributing

We welcome contributions to the project! This guide will help you get started with contributing to `wet-net`.

## Pre-commit Hooks

We use pre-commit hooks to run checks on the code before it is committed. You can install the pre-commit hooks by running the following command in the root of the repository:

```bash
uv run pre-commit install
```

This will ensure that code formatting and linting checks run automatically before each commit. The same checks are also run automatically by the GitHub Actions workflow.

## Release Workflow

We use `uv`, GitHub Actions, and PyPI to manage releases. This guide explains how to cut a new release for `wet-net`.

### Prerequisites

- You should have `uv` installed and configured
- You should be a collaborator on the repository with permission to modify the `pyproject.toml` file and create a new tag.

### Step 1: Bump the Version

Use `uv` to automatically increment the version in `pyproject.toml`:

```bash
# For a patch release (0.2.3 -> 0.2.4)
uv version --bump patch

# For a minor release (0.2.3 -> 0.3.0)
uv version --bump minor

# For a major release (0.2.3 -> 1.0.0)
uv version --bump major
```

This command will:

- Update the `version` field in `pyproject.toml`
- Update `uv.lock` if needed
- Reinstall the package with the new version locally

### Step 2: Commit the Version Change

Stage and commit the version bump:

```bash
git add pyproject.toml uv.lock
git commit -m "Bump version to X.Y.Z"
```

Replace `X.Y.Z` with the actual version number (e.g., `0.2.3`).

### Step 3: Create a new release on GitHub

1. Go to the [Releases](https://github.com/ZachParent/wet-net/releases) page and click the "Draft a new release" button. Select "create a new tag" and name it as the version number (e.g., `v0.2.4`).
2. Fill in the release title and description. The title should be the version number (e.g., `v0.2.4`) and the description should be a short description of the changes. You can generate the changelog automatically.

> [!WARNING]
> Be sure to name the tag with a prefix of `v`, since the GitHub Actions workflow triggers on tags matching the pattern `v*`.

Click the "Publish release" button. This will trigger the GitHub Actions workflow to build the package and publish it to PyPI.

### Best Practices for Releasing

1. **Always test locally** before releasing:

   ```bash
   uv build
   ```

2. **Use semantic versioning**: Follow [SemVer](https://semver.org/) guidelines
   - `MAJOR`: Breaking changes
   - `MINOR`: New features (backward compatible)
   - `PATCH`: Bug fixes (backward compatible)

3. **Write clear, consistent commit messages**: Use descriptive messages like "Bump version to X.Y.Z"

4. **Use the correct tag prefix**: Use the prefix `v` for the tag name (e.g., `v0.2.4`).

5. **Monitor the workflow**: Watch the GitHub Actions workflow to ensure it completes successfully, then check the package on PyPI.

## Additional Resources

- [uv documentation](https://docs.astral.sh/uv/)
- [Git tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging)
- [GitHub Actions](https://docs.github.com/en/actions)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
