from __future__ import annotations

import io
import logging
import lzma
import os
from typing import Any

import pandas as pd
from debian import deb822  # from python-debian
from debian.debian_support import Version
from pandas._libs.missing import NAType  # pyright: ignore[reportPrivateImportUsage]

import osslag.utils.vcs as gh

debian_packages_source_url_template = (
    "https://ftp.debian.org/debian/dists/{release}/main/source/Sources.xz"
)
logger = logging.getLogger(__name__)


def extract_upstream_version(version_string: str) -> str | None:
    """Extract the upstream version from a Debian package version string using
    the official debian.debian_support.Version parser.

    Debian version format: [epoch:]upstream_version[-debian_revision]

    Args:
        version_string: A Debian package version string

    Returns:
        The upstream version, or None if the input is invalid

    Examples:
        >>> extract_upstream_version("1.2.3-4")
        '1.2.3'
        >>> extract_upstream_version("2:1.2.3-4")
        '1.2.3'
        >>> extract_upstream_version("1.2.3")
        '1.2.3'
        >>> extract_upstream_version("1:2.0")
        '2.0'

    """
    if not version_string or not isinstance(version_string, str):
        return None

    try:
        version = Version(version_string.strip())
        upstream = version.upstream_version

        if isinstance(upstream, str):
            # Drop Debian repack/metadata suffixes (e.g., +dfsg, +gitYYYY..., +ds)
            if "+" in upstream:
                upstream = upstream.split("+", 1)[0]

            # Drop prerelease-style suffixes that use '~' (e.g., ~rc1)
            if "~" in upstream:
                upstream = upstream.split("~", 1)[0]

        upstream = upstream.strip() if isinstance(upstream, str) else upstream
        return upstream if upstream else None
    except (ValueError, AttributeError):
        return None


def add_upstream_version_column(
    df: pd.DataFrame, version_column: str, new_column_name: str | None = None
) -> pd.DataFrame:
    """Extract upstream version for each row in a DataFrame and add it as a new column.

    Args:
        df: DataFrame containing version strings
        version_column: Name of the column containing Debian version strings
        new_column_name: Name for the new column (default: "{version_column}_upstream")

    Returns:
        DataFrame with the new upstream version column added

    Raises:
        ValueError: If the specified version_column doesn't exist in the DataFrame

    Examples:
        >>> df = pd.DataFrame(
        ...     {"source": ["pkg1", "pkg2"], "version": ["1.2.3-4", "2:1.0-1"]}
        ... )
        >>> result = add_upstream_version_column(df, "version")
        >>> result["version_upstream"].tolist()
        ['1.2.3', '1.0']

    """
    if version_column not in df.columns:
        raise ValueError(f"Column '{version_column}' not found in DataFrame")

    # Determine the new column name
    if new_column_name is None:
        new_column_name = f"{version_column}_upstream"

    # Apply the extraction function to each row
    df = df.copy()
    df[new_column_name] = df[version_column].apply(extract_upstream_version)

    return df


def add_local_repo_cache_path_column(
    df: pd.DataFrame,
    repo_url_column: str = "homepage",
    cache_dir: str | os.PathLike = "./cache",
    new_column_name: str = "repo_cache_path",
) -> pd.DataFrame:
    """Add a column to the DataFrame with the local repository cache path for each repository URL.

    Args:
        df: DataFrame containing repository URLs
        repo_url_column: Name of the column containing repository URLs
        cache_dir: Base cache directory (default: "./cache")
        new_column_name: Name for the new column (default: "repo_cache_path")

    Returns:
        DataFrame with the new repository cache path column added

    Raises:
        ValueError: If the specified repo_url_column doesn't exist in the DataFrame

    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "source": ["pkg1", "pkg2"],
        ...         "repo_url": [
        ...             "https://github.com/owner/repo1",
        ...             "https://github.com/owner/repo2",
        ...         ],
        ...     }
        ... )
        >>> result = add_local_repo_cache_path_column(
        ...     df, "repo_url", cache_dir="./cache"
        ... )
        >>> "repo_cache_path" in result.columns
        True

    """
    if repo_url_column not in df.columns:
        raise ValueError(f"Column '{repo_url_column}' not found in DataFrame")

    if new_column_name is None:
        new_column_name = "repo_cache_path"

    df = df.copy()

    def _get_cache_path(url: Any) -> str | NAType:
        if (
            url is None
            or (isinstance(url, float) and pd.isna(url))
            or not isinstance(url, str)
        ):
            return pd.NA
        path = gh.construct_repo_local_path(url, cache_dir)
        return str(path) if path is not None else pd.NA

    df[new_column_name] = df[repo_url_column].map(_get_cache_path)
    return df


def filter_github_repos(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include rows with valid GitHub repository URLs.

    Normalizes all GitHub URLs using normalize_https_repo_url and updates the
    homepage column with the cleaned URLs. Rows with invalid or non-GitHub URLs
    are excluded.

    Args:
        df: DataFrame containing a 'homepage' column with repository URLs

    Returns:
        DataFrame filtered to valid GitHub repos with normalized homepage URLs

    """
    if "homepage" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'homepage' column to filter GitHub repositories."
        )

    # First filter to rows containing github.com
    mask = df["homepage"].str.contains("github.com", na=False)
    filtered_df = df[mask].copy()

    # Normalize all GitHub URLs
    normalized = filtered_df["homepage"].apply(gh.normalize_https_repo_url)

    # Extract the normalized URL (or None if invalid)
    filtered_df["homepage"] = normalized.apply(lambda r: r.url)

    # Drop rows where normalization failed (url is None)
    filtered_df = filtered_df[filtered_df["homepage"].notna()]

    # Drop duplicates based on normalized homepage
    github_repos_df = filtered_df.drop_duplicates(subset=["homepage"], keep="first")

    return github_repos_df


def validate_merge_safety(
    df1: pd.DataFrame, df2: pd.DataFrame, merge_key: str = "source"
) -> tuple[bool, list[str]]:
    """Validate if two DataFrames can be safely merged.

    Returns:
        (is_safe, warnings) tuple where is_safe is True if no critical issues found

    """
    warnings = []
    is_safe = True
    logger.info(f"Validating merge safety on key '{merge_key}'")
    # Check if merge key exists in both
    if merge_key not in df1.columns:
        warnings.append(f"Merge key '{merge_key}' missing in first DataFrame")
        is_safe = False
    if merge_key not in df2.columns:
        warnings.append(f"Merge key '{merge_key}' missing in second DataFrame")
        is_safe = False

    if not is_safe:
        return is_safe, warnings

    # Check for duplicates in merge key
    df1_dupes = df1[merge_key].duplicated().sum()
    df2_dupes = df2[merge_key].duplicated().sum()

    if df1_dupes > 0:
        warnings.append(
            f"First DataFrame has {df1_dupes} duplicate '{merge_key}' values"
        )
    if df2_dupes > 0:
        warnings.append(
            f"Second DataFrame has {df2_dupes} duplicate '{merge_key}' values"
        )

    # Check overlapping columns and their dtypes
    common_cols = set(df1.columns) & set(df2.columns) - {merge_key}

    for col in common_cols:
        if df1[col].dtype != df2[col].dtype:
            warnings.append(
                f"Column '{col}' has different dtypes: {df1[col].dtype} vs {df2[col].dtype}"
            )

    # Check merge key overlap
    overlap = set(df1[merge_key]) & set(df2[merge_key])
    overlap_pct = len(overlap) / max(len(df1), len(df2)) * 100

    if overlap_pct < 10:
        warnings.append(f"Low overlap: only {overlap_pct:.1f}% of keys match")

    return is_safe, warnings


def merge_release_packages(
    dfs: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge multiple Debian package DataFrames from different releases on the 'source' column.

    Args:
        dfs: List of DataFrames to merge (expects exactly 2 DataFrames)

    Returns:
        Tuple of (merged_df, unmerged_df) where:
        - merged_df contains rows that matched on 'source' column
        - unmerged_df contains rows that didn't match (from either dataframe)

    """
    if len(dfs) != 2:
        raise ValueError(f"Expected exactly 2 DataFrames, got {len(dfs)}")

    df1, df2 = dfs[0], dfs[1]

    # drop redundant columns before merge
    redundant_columns = set(df1.columns) & set(df2.columns) - {"source"}
    df2 = df2.drop(columns=list(redundant_columns))
    logger.info(
        f"Dropped redundant columns from second DataFrame before merge: {redundant_columns}"
    )

    # Validate merge safety before proceeding
    is_safe, merge_warnings = validate_merge_safety(df1, df2, merge_key="source")

    if not is_safe:
        error_msg = "Cannot safely merge DataFrames:\n  - " + "\n  - ".join(
            merge_warnings
        )
        raise ValueError(error_msg)

    if merge_warnings:
        for warning in merge_warnings:
            logger.warning(f"Merge validation: {warning}")

    # Merge on 'source' column with indicator to track merge status
    merged_df = pd.merge(
        df1, df2, on="source", how="outer", indicator=True, suffixes=("_left", "_right")
    )

    # Separate matched and unmatched rows
    matched = merged_df[merged_df["_merge"] == "both"].copy()
    unmatched = merged_df[merged_df["_merge"].isin(["left_only", "right_only"])].copy()

    # Remove the merge indicator column
    matched = matched.drop(columns=["_merge"])
    unmatched = unmatched.drop(columns=["_merge"])

    # Handle _left/_right column pairs: check if they match, then consolidate
    left_cols = [col for col in matched.columns if col.endswith("_left")]

    for left_col in left_cols:
        base_name = left_col[:-5]  # Remove '_left' suffix
        right_col = base_name + "_right"

        if right_col in matched.columns:
            # Check if columns match (ignoring NaN values)
            mismatches = matched[left_col] != matched[right_col]
            # Account for NaN != NaN being True
            both_nan = matched[left_col].isna() & matched[right_col].isna()
            actual_mismatches = mismatches & ~both_nan

            if actual_mismatches.any():
                mismatch_count = actual_mismatches.sum()
                logger.warning(
                    f"Column '{base_name}' has {mismatch_count} mismatches between releases"
                )

            # Keep left column and rename it, drop right column
            matched = matched.rename(columns={left_col: base_name})
            matched = matched.drop(columns=[right_col])

    matched = matched.rename(columns={"homepage": "upstream_repo_url"})
    unmatched = unmatched.rename(columns={"homepage": "upstream_repo_url"})

    return matched, unmatched


def fetch_packages(release: str) -> pd.DataFrame | None:
    packages_url = debian_packages_source_url_template.format(release=release)
    xz_bytes = gh.fetch_file(packages_url)
    if xz_bytes is None:
        logger.error(f"Failed to fetch Packages.xz for release {release}")
        return None
    # Now xz_bytes contains the raw .xz data of the Sources file
    data = lzma.decompress(xz_bytes)

    # Deb822Source file may contain multiple stanzas
    buf = io.BytesIO(data)
    items = []
    stanza = b""
    for line in buf.readlines():
        if line.strip() == b"":
            if stanza.strip():
                items.append(stanza)
                stanza = b""
        else:
            stanza += line
    if stanza.strip():
        items.append(stanza)
    rows = []
    for st in items:
        try:
            d = deb822.Deb822(st.decode("utf-8", "ignore"))
            rows.append(
                {
                    "source": d.get("Package") or d.get("Source"),
                    f"{release}_version": d.get("Version"),
                    "homepage": d.get("Homepage"),
                    "depends": d.get("Depends"),
                    "maintainer": d.get("Maintainer"),
                }
            )
        except Exception:
            # Log parse failures to error log for visibility
            try:
                logger.error(f"Failed to parse stanza:\n{st.decode('utf-8', 'ignore')}")
            except Exception:
                pass
            continue
    return pd.DataFrame(rows)
