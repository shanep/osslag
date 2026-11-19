# Releasing osslag

The version lives in `Cargo.toml`. Each release goes to crates.io with
`cargo publish`. Publishing a GitHub release builds `osslag` for Linux and
macOS (x86_64 and arm64) and attaches a tarball for each one.
The release workflow fails if the tag does not match the version in
`Cargo.toml`.

## 1. Bump the version

Pick the number with [Semantic Versioning](https://semver.org/):

- **MINOR** (0.1.0 to 0.2.0): a change that breaks the CLI or the parquet files
- **PATCH** (0.1.0 to 0.1.1): new features that keep old caches working, and bug fixes

Cargo treats a 0.x minor bump as breaking, so this holds until 1.0.0. After
that, breaking changes bump MAJOR and new features bump MINOR.

Set `version` in `Cargo.toml`, then update `Cargo.lock` and run the checks:

```bash
cargo update --workspace
make check
git commit -am "Release v0.2.0"
git push origin master
```

## 2. Tag and publish

crates.io never lets you replace a version once it is up, so run the dry run
first. A broken version can only be yanked, which stops new installs from
picking it.

```bash
cargo publish --dry-run --locked
cargo publish --locked
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
gh release create v0.2.0 --generate-notes
```

## 3. Check the release

The `Release` workflow uploads four tarballs named
`osslag-v0.2.0-<target>.tar.gz`. Once it finishes, download the one for your
machine and check the version:

```bash
gh release download v0.2.0 --pattern '*aarch64-apple-darwin*'
tar xzf osslag-v0.2.0-aarch64-apple-darwin.tar.gz
./osslag-v0.2.0-aarch64-apple-darwin/osslag --version
```

If a build failed for a flaky reason (a runner or network hiccup), rerun the
failed job from the Actions tab. A rerun builds the tagged commit again, so if
the code itself needs a fix, fix it on master and release the next patch
version instead.
