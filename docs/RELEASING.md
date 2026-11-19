# Releasing osslag

The version lives in `Cargo.toml`. Publishing a GitHub release builds `osslag`
for Linux and macOS (x86_64 and arm64) and attaches a tarball for each one.
The release workflow fails if the tag does not match the version in
`Cargo.toml`.

## 1. Bump the version

Pick the number with [Semantic Versioning](https://semver.org/):

- **MAJOR** (2.0.0 to 3.0.0): a change that breaks the CLI or the parquet files
- **MINOR** (2.0.0 to 2.1.0): new features that keep old caches working
- **PATCH** (2.0.0 to 2.0.1): bug fixes

Set `version` in `Cargo.toml`, then update `Cargo.lock` and run the checks:

```bash
cargo update --workspace
make check
git commit -am "Release v2.1.0"
git push origin master
```

## 2. Tag and publish

```bash
git tag -a v2.1.0 -m "Release v2.1.0"
git push origin v2.1.0
gh release create v2.1.0 --generate-notes
```

## 3. Check the release

The `Release` workflow uploads four tarballs named
`osslag-v2.1.0-<target>.tar.gz`. Once it finishes, download the one for your
machine and check the version:

```bash
gh release download v2.1.0 --pattern '*aarch64-apple-darwin*'
tar xzf osslag-v2.1.0-aarch64-apple-darwin.tar.gz
./osslag-v2.1.0-aarch64-apple-darwin/osslag --version
```

If a build failed for a flaky reason (a runner or network hiccup), rerun the
failed job from the Actions tab. A rerun builds the tagged commit again, so if
the code itself needs a fix, fix it on master and release the next patch
version instead.
