"""Score the parity sample with the Python MALTA code, for scripts/parity_run.sh.

usage: parity_score.py PY_CACHE RS_CACHE EVAL_END

Reads the merged packages and commits from PY_CACHE, and the pull requests and metadata from
RS_CACHE so both scorers see the same GitHub data. Writes PY_CACHE/debian_malta_scores.parquet.
"""

import sys
from datetime import datetime, timezone

import pandas as pd

from osslag.metrics.malta import score_repos
from osslag.utils.vcs import label_trivial_commits


def main(py: str, rs: str, eval_end: str) -> None:
    merged = pd.read_parquet(f"{py}/debian_merged_releases_packages.parquet")
    packages: list[tuple[str, str]] = list(dict.fromkeys(zip(merged["source"], merged["upstream_repo_url"])))
    commits = label_trivial_commits(pd.read_parquet(f"{py}/debian_all_upstream_commits.parquet"))
    prs = pd.read_parquet(f"{rs}/debian_all_upstream_pull_requests.parquet")
    meta = pd.read_parquet(f"{rs}/debian_all_upstream_metadata.parquet")
    # score_repos reads stars, forks, ... and quietly uses 0 when they are missing, but the
    # metadata step writes stargazers_count, forks_count, ... The Rust scorer takes either.
    meta = meta.rename(columns={f"{c}_count": c for c in ("forks", "watchers", "open_issues")})
    meta = meta.rename(columns={"stargazers_count": "stars"})
    end = datetime.strptime(eval_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    result = score_repos(packages, commits, prs, meta, end, show_progress=False)
    result = result.sort_values("source", kind="stable").reset_index(drop=True)
    result.to_parquet(f"{py}/debian_malta_scores.parquet", index=False)


# The process pool in score_repos re-imports this file, so the work must not run on import.
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
