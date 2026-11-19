# OSS-Lag: Open Source Software Lag Dataset

This repository builds a dataset that measures technical lag and abandonment of
open source packages in Linux distributions. It takes the Debian packages whose
upstream lives on GitHub, clones each upstream, and collects its commits, pull
requests and repository metadata. Then it scores each package with MALTA.

The scoring follows the paper "MALTA: Maintenance-Aware Technical Lag
Estimation to Address Software Abandonment" (Panter and Eisty).

## Install

Install it from [crates.io](https://crates.io/crates/osslag) with a Rust
toolchain (1.85 or newer):

```bash
cargo install osslag --locked
```

Or download the tarball for your platform from the
[releases page](https://github.com/shanep/osslag/releases) and put `osslag`
somewhere on your `PATH`. Builds are there for Linux and macOS, on x86_64 and
arm64.

## Setup

Copy `env.template` to `.env` and fill in your GitHub token. Without a token
GitHub allows 60 requests an hour, which runs out after a couple of packages.

| Variable | What it does | Default |
|---|---|---|
| `GITHUB_TOKEN` | GitHub API token | none |
| `GITHUB_USERNAME` | Used with the token for git clones | none |
| `CACHE_DIR` | Where the pipeline files go | `./cache` |
| `REPOS_CACHE_DIR` | Where the clones go | `$CACHE_DIR/repos` |
| `MAX_WORKERS` | Parallel workers for steps 5 to 8 | 4 |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` or `ERROR` | `INFO` |
| `LOG_FILE` | Log file, rotated at 5 MB with 3 backups | `osslag.log` |

Variables already set in the shell win over `.env`.

## Build the dataset

Run every step for two Debian releases:

```bash
osslag dataset run --release bookworm --release trixie
```

Each step skips work that is already in the cache, so you can stop a run and
start it again. Pass `--force` to redo everything. Every step also has its own
subcommand (`osslag dataset --help` lists them).

Or watch it run in a full-screen view. Press r to run every step, 1 to 8 to run
one step, s to stop and q to quit:

```bash
osslag dataset tui --release bookworm --release trixie
```

## Score the packages

```bash
osslag score --eval-end 2026-11-01
```

This writes `debian_malta_scores.parquet` to the cache, one row per package.
Use `--commits-since` on `dataset run` only if the date is at or before the
start of the MALTA baseline window (42 months before `--eval-end`).

## Changes from the Python version

Versions 1.x were written in Python. Version 2 is a Rust rewrite that reads
and writes the same parquet files, so old caches still work, and gives the same
MALTA scores. A few things changed on purpose:

1. A failed task leaves one marker, `{checkpoint_dir}/{source}.failed`, and
   every step skips it next time. Python wrote a marker it never read, so
   failures were retried on every run.
2. `dataset run` takes `--max-workers` and `--repos-cache` and passes them to
   steps 5 to 8.
3. Scores are sorted by `source`.
4. A closed pull request with neither `merged_at` nor `closed_at` gives an
   error row for that package.
5. Package files no longer have the `__index_level_0__` column.
6. When the GitHub rate limit runs out, requests wait for the reset instead of
   failing.

## Development

```bash
git config core.hooksPath .githooks   # run make check before every push
make check                            # fmt, clippy and tests
make build                            # target/release/osslag
```

`tests/fixtures/parity/` holds output from the last Python version. The tests
in `tests/parity.rs` check that the Rust code still matches it exactly.

See [docs/RELEASING.md](docs/RELEASING.md) to cut a release.
