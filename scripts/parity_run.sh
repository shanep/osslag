#!/usr/bin/env bash
# Run the Python and Rust pipelines side by side on real Debian data and compare every file.
#
# usage: scripts/parity_run.sh [-n SAMPLE] [-s SEED] [-e EVAL_END] [-r RELEASES] SCRATCH_DIR
#
# Steps 1 to 4 run in full on both sides. Then both use the same random sample of merged
# packages for steps 5 to 8. GitHub is only called by the Rust side (clone check, metadata,
# pull requests), since stars and PRs can change between two fetches. Both scorers read the
# same metadata and pull request files. Exits 1 if any comparison differs.
set -euo pipefail

SAMPLE=25
SEED=42
EVAL_END=2026-11-01
RELEASES="bookworm trixie"
while getopts "n:s:e:r:h" opt; do
    case $opt in
        n) SAMPLE=$OPTARG ;;
        s) SEED=$OPTARG ;;
        e) EVAL_END=$OPTARG ;;
        r) RELEASES=${OPTARG//,/ } ;;
        *) sed -n '2,10p' "$0"; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[ $# -eq 1 ] || { sed -n '2,10p' "$0"; exit 2; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH=$(mkdir -p "$1" && cd "$1" && pwd)
PY=$SCRATCH/py
RS=$SCRATCH/rs
REPOS=$SCRATCH/repos
mkdir -p "$PY" "$RS" "$REPOS"

# Both CLIs let these variables win over their flags, and both load .env, which only fills
# variables that are unset. So set them here for each side rather than unsetting them.
export REPOS_CACHE_DIR=$REPOS MAX_WORKERS=4 LOG_FILE=$SCRATCH/osslag.log

cd "$ROOT"
cargo build --release --quiet --bin osslag --example parquet_diff
RUST=$ROOT/target/release/osslag
DIFF=$ROOT/target/release/examples/parquet_diff

py() { CACHE_DIR=$PY uv run --quiet osslag "$@"; }
rs() { CACHE_DIR=$RS "$RUST" "$@"; }

# shellcheck disable=SC2086 # RELEASES is a word list on purpose
for side in py rs; do
    cache=$SCRATCH/$side
    echo "== $side: steps 1 to 4"
    $side dataset fetch-packages debian $RELEASES --cache "$cache"
    $side dataset filter-debian-github-repos debian $RELEASES --cache "$cache"
    $side dataset extract-upstream-versions debian $RELEASES --cache "$cache"
    $side dataset merge-releases debian $RELEASES --cache "$cache"
done

FAILED=0
same() {
    local name=$1
    shift
    echo "== compare $name"
    "$DIFF" "$PY/$name" "$RS/$name" --ignore __index_level_0__ "$@" || FAILED=1
}

for rel in $RELEASES; do
    same "debian_${rel}_all_packages.parquet" --key source
    same "debian_${rel}_filtered_packages.parquet" --key source
    same "debian_${rel}_packages_with_upstream_versions.parquet" --key source
done
same debian_merged_releases_packages.parquet --key source
same debian_dropped_after_merge.parquet --key source

echo "== sample $SAMPLE packages (seed $SEED)"
uv run --quiet python - "$PY/debian_merged_releases_packages.parquet" "$SAMPLE" "$SEED" "$RS" <<'EOF'
import shutil
import sys

import pandas as pd

merged, n, seed, rust_cache = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
df = pd.read_parquet(merged)
df.sample(n=min(n, len(df)), random_state=seed).sort_values("source").to_parquet(merged, index=False)
shutil.copy(merged, f"{rust_cache}/debian_merged_releases_packages.parquet")
EOF

echo "== rs: steps 5 to 8"
rs dataset clone-upstream-repos debian --cache "$RS" --repos-cache "$REPOS"
rs dataset load-commits-into-dataframe debian --cache "$RS" --repos-cache "$REPOS"
rs dataset all-github-metadata --distro debian --cache "$RS"
rs dataset all-github-pull-requests --distro debian --cache "$RS"

echo "== py: step 6 on the same clones"
py dataset load-commits-into-dataframe debian --cache "$PY" --repos-cache "$REPOS"
same debian_all_upstream_commits.parquet --key source,hash

echo "== score with eval end $EVAL_END"
rs score --distro debian --eval-end "$EVAL_END" --cache "$RS"
uv run --quiet python scripts/parity_score.py "$PY" "$RS" "$EVAL_END"
same debian_malta_scores.parquet --key source

if [ "$FAILED" -eq 0 ]; then
    echo "parity: every file matches"
else
    echo "parity: differences found (see above)"
fi
exit "$FAILED"
