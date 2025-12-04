# Release Workflow Tutorial

This guide explains how to cut a new release for `wet-net` using `uv`, GitHub, and PyPI.

## Prerequisites

- `uv` installed and configured
- Git repository with write access
- GitHub Actions workflow configured (`.github/workflows/release.yml`)
- PyPI trusted publishing configured for the GitHub repository

## Release Workflow

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

### Step 3: Create a Git Tag

Create an annotated tag pointing to the version bump commit:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
```

**Important:** The tag name must start with `v` (e.g., `v0.2.3`) because the GitHub Actions workflow triggers on tags matching the pattern `v*`.

### Step 4: Push Commit and Tag

```bash
git push origin main --follow-tags
```

### Step 5: Verify the Release

After pushing the tag:

1. **GitHub Actions**: Check the Actions tab in GitHub to see the workflow running
2. **PyPI**: Once the workflow completes, verify the package appears on PyPI
3. **Installation**: Test installing the new version:

   ```bash
   uvx wet-net --help
   ```

## Complete Example

Here's a complete example for releasing version `0.2.4`:

```bash
# 1. Bump version
uv version --bump patch

# 2. Verify the version was updated
grep '^version =' pyproject.toml

# 3. Commit the change
git add pyproject.toml uv.lock
git commit -m "Bump version to 0.2.4"

# 4. Create tag
git tag -a v0.2.4 -m "v0.2.4"

# 5. Push everything
git push origin main && git push origin v0.2.4
```

## Best Practices

1. **Always test locally** before releasing:

   ```bash
   uv build
   ```

2. **Use semantic versioning**: Follow [SemVer](https://semver.org/) guidelines
   - `MAJOR`: Breaking changes
   - `MINOR`: New features (backward compatible)
   - `PATCH`: Bug fixes (backward compatible)

3. **Write clear commit messages**: Use descriptive messages like "Bump version to X.Y.Z"

4. **Verify before pushing**: Check that the tag points to the correct commit:

   ```bash
   git show vX.Y.Z --format="%H %s" --no-patch
   ```

5. **Monitor the workflow**: Watch the GitHub Actions workflow to ensure it completes successfully

## Additional Resources

- [uv documentation](https://docs.astral.sh/uv/)
- [Git tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging)
- [GitHub Actions](https://docs.github.com/en/actions)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
