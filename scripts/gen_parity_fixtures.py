"""Generate golden fixtures that pin the Rust port to the Python behavior.

Run once with the Python package installed (uv run python scripts/gen_parity_fixtures.py).
The outputs in tests/fixtures/parity/ are committed and read by the Rust tests.
"""

from __future__ import annotations

import ast
import json
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from osslag.distro.debian import extract_upstream_version
from osslag.metrics.malta import score_repos
from osslag.metrics.pvac import categorize_development_activity, lookup_category
from osslag.utils.vcs import label_trivial_commits, normalize_https_repo_url

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "parity"
EVAL_END = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_string_literals() -> list[str]:
    """Every string literal in the Python tests, so their inputs are all covered."""
    found: set[str] = set()
    for path in (ROOT / "tests").rglob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 200:
                found.add(node.value)
    return sorted(found)


EXTRA_VERSIONS = [
    "1.2.3", "0:1.2.3", "1:2.0.0-rc.1+build.5", "01.02.03", "1.2", "1.2p3", "1.2pl4", "1.2.3-01",
    "2:7.4.052-1ubuntu3.1", "1.0~rc1-2", "1.10.0+git20230101-1", "8.8.1+ds+~cs25.17.7", "1.2-alpha-3",
    "", "   ", "abc", "1", "1.2.3.4", "v1.2.3", "20230101", "3.0.0-beta", "0.0.0", "10.20.30",
]
EXTRA_URLS = [
    "https://github.com/foo/bar", "http://github.com/foo/bar.git", "https://github.com/foo/bar/",
    "https://github.com/foo/bar/tree/main", "https://github.com/foo/bar#readme", "git@github.com:foo/bar.git",
    "https://gitlab.com/foo/bar", "git://github.com/foo/bar", "https://user@github.com/foo/bar",
    "  https://github.com/Foo/Bar.js  ", "https://github.com/foo/bar.git/", "https://github.com/foo",
    "", "ftp://github.com/foo/bar",
]


def gen_strings() -> None:
    inputs = sorted(set(test_string_literals()) | set(EXTRA_VERSIONS) | set(EXTRA_URLS))
    rows = []
    for s in inputs:
        cat = lookup_category(s)
        norm = normalize_https_repo_url(s)
        rows.append(
            {
                "input": s,
                "upstream_version": extract_upstream_version(s),
                "pvac": dict(cat),
                "normalized_url": norm.url,
                "normalize_error": norm.error,
            }
        )
    pairs = []
    for a in EXTRA_VERSIONS:
        for b in EXTRA_VERSIONS:
            pairs.append({"a": a, "b": b, "activity": categorize_development_activity(a, b)})
    (OUT / "strings.json").write_text(json.dumps({"cases": rows, "activity": pairs}, indent=1) + "\n")


def gen_trivial() -> None:
    files = [[], ["README.md"], ["docs/a.MD", "b.md"], ["a.md", "src/x.rs"], ["Makefile"], ["notes.markdown"], [".md"]]
    df = pd.DataFrame({"files": files})
    labeled = label_trivial_commits(df)
    (OUT / "trivial.json").write_text(
        json.dumps([{"files": f, "is_trivial": bool(t)} for f, t in zip(files, labeled["is_trivial"])], indent=1)
        + "\n"
    )


def rand_time(rng: random.Random, start: datetime, end: datetime) -> datetime:
    span = int((end - start).total_seconds() * 1_000_000)
    return start + timedelta(microseconds=rng.randrange(span))


def gen_malta() -> None:
    rng = random.Random(20261118)
    oldest = EVAL_END - timedelta(days=365 * 6)
    newest = EVAL_END + timedelta(days=30)
    commits: list[dict[str, object]] = []
    prs: list[dict[str, object]] = []
    meta: list[dict[str, object]] = []
    packages: list[tuple[str, str]] = []
    for i in range(300):
        url = f"https://github.com/owner{i}/repo{i}"
        packages.append((f"pkg{i}", url))
        kind = i % 10
        n_commits = 0 if kind == 0 else rng.choice([1, 3, 10, 50, 200])
        for _ in range(n_commits):
            commits.append({"repo_url": url, "date": rand_time(rng, oldest, newest), "is_trivial": rng.random() < 0.2})
        n_prs = 0 if kind in (0, 1) else rng.choice([1, 2, 5, 20, 60])
        for _ in range(n_prs):
            created = rand_time(rng, oldest, EVAL_END)
            state = rng.choice(["closed", "closed", "open", "draft"] if kind == 9 else ["closed", "closed", "open"])
            closed = merged = None
            if state == "closed":
                closed = created + timedelta(seconds=rng.randrange(0, 86400 * 400))
                if rng.random() < 0.6:
                    merged = closed - timedelta(seconds=rng.randrange(0, 3600))
            prs.append({"repo_url": url, "created_at": created, "closed_at": closed, "merged_at": merged, "state": state})
        if kind != 2:
            meta.append(
                {
                    "repo_url": url,
                    "stars": rng.choice([0, 1, 50, 5000, 50000]),
                    "forks": rng.choice([0, 3, 900, 20000]),
                    "watchers": rng.choice([0, 10, 400, 12000]),
                    "open_issues": rng.choice([0, 5, 300, 15000]),
                    "archived": kind == 3,
                }
            )
    # A package whose repo shows up twice in meta: only the first row counts.
    meta.append({"repo_url": "https://github.com/owner5/repo5", "stars": 1, "forks": 1, "watchers": 1, "open_issues": 1, "archived": True})
    # A package with no data anywhere.
    packages.append(("ghost", "https://github.com/nobody/ghost"))

    commits_df = pd.DataFrame(commits, columns=["repo_url", "date", "is_trivial"])
    prs_df = pd.DataFrame(prs, columns=["repo_url", "created_at", "closed_at", "merged_at", "state"])
    meta_df = pd.DataFrame(meta)
    for col in ("created_at", "closed_at", "merged_at"):
        prs_df[col] = pd.to_datetime(prs_df[col], utc=True)
    commits_df["date"] = pd.to_datetime(commits_df["date"], utc=True)

    result = score_repos(packages, commits_df, prs_df, meta_df, EVAL_END, n_workers=4, show_progress=False)
    result = result.sort_values("source", kind="stable").reset_index(drop=True)

    commits_df.to_parquet(OUT / "malta_commits.parquet", index=False)
    prs_df.to_parquet(OUT / "malta_prs.parquet", index=False)
    meta_df.to_parquet(OUT / "malta_meta.parquet", index=False)
    pd.DataFrame(packages, columns=["source", "repo_url"]).to_parquet(OUT / "malta_packages.parquet", index=False)
    result.to_parquet(OUT / "malta_expected.parquet", index=False)
    (OUT / "malta_eval_end.txt").write_text(EVAL_END.isoformat() + "\n")


DEBIAN_SOURCES = {
    "bookworm": (
        b"Package: alpha\nVersion: 1.2.3-1\nHomepage: https://github.com/o/alpha\nMaintainer: A <a@x.org>\n\n"
        b"Package: beta\nVersion: 2:4.5-2\nHomepage: http://github.com/o/beta.git\nDepends: libc6,\n libz\n\n"
        b"  \t\n"
        b"Package: gamma\nVersion: 1.0~rc1-1\nHomepage: https://github.com/o/gamma/tree/main\r\n\n"
        b"Package: delta\nVersion: 3.0+dfsg-1\nHomepage: https://gitlab.com/o/delta\n\n"
        b"Package: epsilon\nVersion: 1.0-1\n\n"
        b"Package: zeta\nVersion: 1.0-1\nHomepage: https://githubXcom/o/zeta\n\n"
        b"Package: eta\nVersion: 1.0-1\nHomepage: git@github.com:o/eta.git\n\n"
        b"Package: theta\nVersion: 5.0-1\nHomepage: https://github.com/o/alpha#readme\n\n"
        b"# comment\nSource: iota\nVersion: a:b\nHomepage: https://github.com/o/iota\nMaintainer: Bj\xf6rn <b@x>\n\n"
        b"Package: kappa\nVersion: 7.1-1\nHomepage: https://github.com/o/kappa\n\n"
        b"Package: kappa\nVersion: 7.2-1\nHomepage: https://github.com/o/kappa2\n\n"
        b"Package: only-old\nVersion: 0.1-1\nHomepage: https://github.com/o/only-old\n\n"
        b"Package: lambda\nVersion: 1:2.0-3\nHomepage: HTTPS://github.com/o/lambda\n"
    ),
    "trixie": (
        b"Package: kappa\nVersion: 7.3-1\nHomepage: https://github.com/o/kappa\n\n"
        b"Package: alpha\nVersion: 1.3.0-1\nHomepage: https://github.com/o/alpha\nMaintainer: New <n@x.org>\n\n"
        b"Package: beta\nVersion: 2:4.6-1\nHomepage: https://github.com/o/beta\n\n"
        b"Package: gamma\nVersion: 1.0-1\nHomepage: https://github.com/o/gamma\n\n"
        b"Package: iota\nVersion: 2.0-1\nHomepage: https://github.com/o/iota\n\n"
        b"Package: only-new\nVersion: 9.9-1\nHomepage: https://github.com/o/only-new\n\n"
    ),
}


def gen_debian() -> None:
    """Run pipeline steps 1-4 the way the CLI does, on two small Sources files with edge cases."""
    import lzma
    import tempfile

    from osslag.distro import debian

    os.environ.pop("REPOS_CACHE_DIR", None)
    tables = []
    with tempfile.TemporaryDirectory() as empty_cache:
        for rel, text in DEBIAN_SOURCES.items():
            xz = lzma.compress(text)
            (OUT / f"debian_{rel}_Sources.xz").write_bytes(xz)
            debian.gh.fetch_file = lambda url, xz=xz: xz  # type: ignore[assignment]
            all_df = debian.fetch_packages(rel)
            assert all_df is not None
            all_df.to_parquet(OUT / f"debian_{rel}_all_packages.parquet", index=False)
            filtered = debian.filter_github_repos(all_df)
            filtered = debian.add_local_repo_cache_path_column(filtered, cache_dir=empty_cache)
            filtered.reset_index(drop=True, inplace=True)
            filtered.to_parquet(OUT / f"debian_{rel}_filtered_packages.parquet", index=False)
            col = f"{rel}_upstream_version"
            versions = debian.add_upstream_version_column(filtered, f"{rel}_version", new_column_name=col)
            versions.dropna(subset=[col], inplace=True)
            versions.reset_index(drop=True, inplace=True)
            versions.to_parquet(OUT / f"debian_{rel}_packages_with_upstream_versions.parquet", index=False)
            tables.append(versions)
    merged, dropped = debian.merge_release_packages(tables)
    merged.reset_index(drop=True).to_parquet(OUT / "debian_merged_releases_packages.parquet", index=False)
    dropped.reset_index(drop=True).to_parquet(OUT / "debian_dropped_after_merge.parquet", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gen_strings()
    gen_trivial()
    gen_malta()
    gen_debian()
    print(f"wrote fixtures to {OUT}")
