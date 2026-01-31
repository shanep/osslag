from __future__ import annotations

import logging
import os
import pathlib
import re
import shutil
from datetime import datetime
from typing import TYPE_CHECKING, List, NamedTuple, cast

if TYPE_CHECKING:
    from pygit2.enums import DiffOption

import pandas as pd
import pygit2
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present
logger = logging.getLogger(__name__)


class NormalizeRepoResult(NamedTuple):
    url: str | None
    error: str | None


class RepoOwnerName(NamedTuple):
    owner: str | None
    name: str | None


class CloneResult(NamedTuple):
    success: bool
    error: str | None


def normalize_https_repo_url(url: str) -> NormalizeRepoResult:
    if not url or not isinstance(url, str):
        return NormalizeRepoResult(None, "Invalid URL: None or not a string")
    url = url.strip()

    # Strip fragment (e.g., #readme) from URLs
    if "#" in url:
        url = url.split("#")[0]

    # Convert http:// to https://
    if url.startswith("http://"):
        url = "https://" + url[7:]

    # Strip .git suffix early if present (for all protocols)
    url_for_matching = url
    if url.endswith(".git"):
        url_for_matching = url[:-4]

    # Strip trailing slash
    if url_for_matching.endswith("/"):
        url_for_matching = url_for_matching[:-1]

    # GitHub SSH pattern - return None for SSH URLs
    github_ssh_pattern = re.compile(r"^git@github\.com:([\w.-]+)/([\w.-]+)$")
    match = github_ssh_pattern.match(url_for_matching)
    if match:
        return NormalizeRepoResult(None, None)

    # GitHub HTTPS pattern - clean up garbage
    github_https_pattern = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+)")
    match = github_https_pattern.match(url_for_matching)
    if match:
        # Reconstruct clean URL without .git
        clean_url = f"https://github.com/{match.group(1)}/{match.group(2)}"
        return NormalizeRepoResult(clean_url, None)

    return NormalizeRepoResult(
        None, "URL does not match expected git repository patterns"
    )


def ensure_dir(p: str | os.PathLike) -> pathlib.Path:
    p = pathlib.Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_file(url) -> bytes:
    """Fetch a file from a URL and return its content as bytes.

    Args:
        url: The URL of the file to fetch.
        Returns: The content of the file as bytes.

    """
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        chunk_size = 1024
        content = bytearray()
        for chunk in r.iter_content(chunk_size=chunk_size):
            content.extend(chunk)
    return bytes(content)


def extract_owner_name_repo(repo_url: str) -> RepoOwnerName:
    """Extract owner and repository name from a GitHub token string. Expects
    a URL that has been normalized by normalize_repo_url.

    Args:
        repo_url: The GitHub repository URL.

    Returns:
        A tuple containing the owner and repository name.

    """
    repo_url_normalized = normalize_https_repo_url(repo_url)
    if repo_url_normalized and repo_url_normalized.url is not None:
        parts = repo_url_normalized.url.split("/")
        if len(parts) < 2:
            logger.error(
                "Could not extract owner/repo from URL: call normalize_repo_url first"
            )
            return RepoOwnerName(None, None)
        owner = parts[-2]
        repo = parts[-1]
        return RepoOwnerName(owner, repo)
    return RepoOwnerName(None, None)


def clone_repo(
    repo_url: str,
    dest_dir: str | os.PathLike,
    include_tags: bool = True,
    branch: str | None = None,
    server_connect_timeout: int = 10000,
    server_timeout: int = 300000,
) -> CloneResult:
    """Clone a Git repository using PyGithub to get the full repository with history. This function
    is written to run currently in multiple processes (since pygit2 is not thread-safe) and
    python 's multiprocessing module is recommended for parallelism.

    Args:
        repo_url: The URL of the Git repository (GitHub URLs).
        dest_dir: Optional destination directory. If not provided, a temporary directory is used.
        include_tags: If True, fetches tags during clone.
        server_connect_timeout: Connection timeout in milliseconds, default 10000 (10 seconds).
        server_timeout: General server timeout in milliseconds, default 300000 (300 seconds).

    Returns:
        CloneResult with success flag and error message if any.

    """
    # Validate input
    if not repo_url or not isinstance(repo_url, str):
        return CloneResult(False, "Invalid URL: None or not a string")

    dest_path = pathlib.Path(dest_dir)
    failed_marker = dest_path.parent / f"{dest_path.name}.failed"

    # Normalize the URL
    sanitize_url = normalize_https_repo_url(repo_url)
    if sanitize_url.url is None:
        error_msg = sanitize_url.error or "Failed to normalize repository URL"
        logger.warning(f"Failed to normalize URL {repo_url}: {error_msg}")
        try:
            failed_marker.write_text(f"normalize: {error_msg}")
        except Exception:
            pass
        return CloneResult(False, error_msg)

    repo_url = sanitize_url.url
    owner, repo = extract_owner_name_repo(repo_url)
    if owner is None or repo is None:
        error_msg = "Failed to extract owner/repo from URL"
        logger.warning(f"Failed to extract owner/repo from {repo_url}")
        try:
            failed_marker.write_text(f"extract: {error_msg}")
        except Exception:
            pass
        return CloneResult(False, error_msg)

    # Check if local checkout exists
    git_dir = pathlib.Path(dest_dir) / ".git"
    if git_dir.exists():
        return CloneResult(True, None)

    # Check if repo exists on GitHub (lazy import to avoid circular dependency)
    from osslag.utils.github_helper import gh_check_repo_exists

    repo_exists_result = gh_check_repo_exists(owner, repo)
    if not repo_exists_result.success:
        error_msg = repo_exists_result.error or "Repository not found on GitHub"
        try:
            failed_marker.write_text(f"gh_check: {error_msg} url: {repo_url}")
        except Exception:
            pass
        # Pass through rate limit info
        return CloneResult(
            success=False,
            error=error_msg,
        )

    pygit2.settings.server_connect_timeout = server_connect_timeout
    pygit2.settings.server_timeout = server_timeout

    # Clone using pygit2
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        github_username = os.getenv("GITHUB_USERNAME")
        if github_token and github_username:
            callbacks = pygit2.RemoteCallbacks(
                pygit2.UserPass(github_username, github_token)
            )
        else:
            callbacks = None

        repo_obj = pygit2.clone_repository(
            sanitize_url.url, dest_dir, callbacks=callbacks, bare=False
        )

        # Checkout specific branch if requested
        if branch:
            repo_obj.revparse_single(branch)
            repo_obj.checkout(branch)

        # Fetch tags if requested
        if include_tags:
            remote = repo_obj.remotes["origin"]
            remote.fetch()
        return CloneResult(True, None)

    except pygit2.GitError as e:
        # Cleanup partial clone on failure
        dest_path = pathlib.Path(dest_dir)
        if dest_path.exists():
            try:
                shutil.rmtree(dest_path)
            except Exception:
                pass
        error_msg = f"pygit2 error: {str(e)}"
        try:
            failed_marker.write_text(f"clone: {error_msg}")
        except Exception:
            pass
        return CloneResult(False, error_msg)
    except Exception as e:
        # Cleanup partial clone on failure
        dest_path = pathlib.Path(dest_dir)
        if dest_path.exists():
            try:
                shutil.rmtree(dest_path)
            except Exception:
                pass
        error_msg = f"General error: {str(e)}"
        try:
            failed_marker.write_text(f"clone: {error_msg}")
        except Exception:
            pass
        return CloneResult(False, error_msg)


def construct_repo_local_path(
    repo_url: str, cache_dir: str | os.PathLike = "./cache", must_exist: bool = True
) -> pathlib.Path | None:
    """Get the local path for a cloned repository based on its URL and cache directory.

    Args:
        repo_url: The URL of the repository
        cache_dir: Base cache directory (default: "./cache")
        must_exist: If True, returns None if the path doesn't exist (default: True)

    Returns:
        The local path as a Path object, or None if the URL is invalid
        (or if must_exist=True and path doesn't exist)

    Examples:
        >>> construct_repo_local_path("https://github.com/owner/repo", "./cache")
        None  # if not cloned yet
        >>> construct_repo_local_path(
        ...     "https://github.com/owner/repo", "./cache", must_exist=False
        ... )
        PosixPath('./cache/owner--repo')

    """
    sanitized_url = normalize_https_repo_url(repo_url)
    if sanitized_url.url is None:
        return None

    repo_owner = extract_owner_name_repo(sanitized_url.url)
    if repo_owner.owner is None or repo_owner.name is None:
        return None

    REPOS_CACHE_DIR = os.getenv("REPOS_CACHE_DIR") or str(cache_dir)
    local_repo_path = (
        pathlib.Path(REPOS_CACHE_DIR) / f"{repo_owner.owner}--{repo_owner.name}"
    )
    if must_exist and not local_repo_path.exists():
        return None
    if local_repo_path.exists():
        local_repo_path = local_repo_path.resolve()
    return local_repo_path


def label_trivial_commits(
    commits_df: pd.DataFrame,
    files_column: str = "files",
    label_column: str = "is_trivial",
    cache_dir: os.PathLike | None = None,
    cache_name: str = "commits_with_trivial_labels.parquet",
) -> pd.DataFrame:
    """Add a boolean column marking commits that only changed README.md.

    A commit is marked True when the files list contains exactly one entry
    and that entry's basename is README.md (case-insensitive, any path).

    Args:
        commits_df: DataFrame returned by get_commits_between_tags (expects a 'files' column).
        files_column: Name of the column containing file path lists.
        label_column: Name of the output boolean column to create.
        cache_dir: Optional cache directory
        cache_name: Optional cache file name (defaults to 'commits_with_trivial_labels.parquet' if cache_dir is set)

    Returns:
        The same DataFrame with an added boolean column.

    """
    if files_column not in commits_df.columns:
        logger.error(f"Missing '{files_column}' column; cannot label trivial commits")
        return commits_df

    def _is_trivial(files: List[str]) -> bool:
        # Empty files list is considered trivial
        if len(files) == 0:
            return True

        # Documentation-only change if all files are .md files
        rval = all(
            isinstance(f, str) and pathlib.PurePosixPath(f).suffix.lower() == ".md"
            for f in files
        )
        return rval

    commits_df[label_column] = commits_df[files_column].apply(_is_trivial)
    commits_df = commits_df.reset_index(drop=True)
    if cache_dir is not None:
        cache_path = pathlib.Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / cache_name
        commits_df.to_parquet(cache_file)
    return commits_df


def load_commits(
    repo_path: str | os.PathLike,
    branch: str | None = None,
    include_files: bool = True,
    since: datetime | None = None,
) -> pd.DataFrame:
    """Retrieve commits from a local Git repository and return as a pandas DataFrame.

    Args:
        repo_path: Path to the local Git repository directory.
        branch: Branch name to walk from. If None, uses HEAD.
        include_files: If True, include list of changed files per commit. Default True.
        since: Only include commits after this date. If None, defaults to 4 years ago.
               Pass a very old date (e.g., datetime(1970, 1, 1)) to get all commits.

    Returns:
        A pandas DataFrame with columns: 'hash', 'author', 'email', 'message',
        'timestamp', 'date', and optionally 'files'.

    Examples:
        >>> df = load_commits("/path/to/repo")
        >>> df = load_commits("/path/to/repo", branch="main")
        >>> df = load_commits("/path/to/repo", since=datetime(2020, 1, 1))

    Raises:
        FileNotFoundError: If repo path doesn't exist or is not a git repository.
        ValueError: If the specified branch is not found or repo has no HEAD.
        RuntimeError: For other git-related errors.

    """
    # Default to 4 years ago if not specified
    if since is None:
        since = datetime.now() - relativedelta(years=4)
    repo_path = pathlib.Path(repo_path)

    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    git_dir = repo_path / ".git"
    if not git_dir.exists():
        raise FileNotFoundError(
            f"Not a Git repository (missing .git directory): {repo_path}"
        )

    try:
        repo = pygit2.Repository(str(repo_path))

        # Determine starting point
        if branch:
            try:
                ref = repo.references[f"refs/heads/{branch}"]
                start_id = ref.peel(pygit2.Commit).id
            except KeyError:
                # Try remote branch
                try:
                    ref = repo.references[f"refs/remotes/origin/{branch}"]
                    start_id = ref.peel(pygit2.Commit).id
                except KeyError as e:
                    raise ValueError(
                        f"Branch '{branch}' not found in {repo_path}"
                    ) from e
        else:
            try:
                start_id = repo.head.peel(pygit2.Commit).id
            except pygit2.GitError as e:
                raise ValueError(
                    f"Repository has no HEAD (empty repository?): {repo_path}"
                ) from e

        def _changed_paths(c: pygit2.Commit) -> list[str]:
            """Return list of paths touched by the commit (optimized, skips merge commits)."""
            # Skip merge commits (more than one parent)
            if len(c.parents) > 1:
                return []
            try:
                # Diff flags for speed:
                # - SKIP_BINARY_CHECK: Don't examine binary file contents
                # Cast needed because pygit2 exports int constants but stubs expect DiffOption enum
                flags = cast("DiffOption", pygit2.GIT_DIFF_SKIP_BINARY_CHECK)

                if c.parents:
                    # Diff from parent to commit (shows what changed in this commit)
                    # context_lines=0 skips computing context around changes
                    diff = c.parents[0].tree.diff_to_tree(
                        c.tree, flags=flags, context_lines=0
                    )
                else:
                    # Initial commit - diff against empty tree
                    diff = c.tree.diff_to_tree(flags=flags, context_lines=0)

                # Extract paths directly with list comprehension
                # new_file.path is set for adds/modifies, old_file.path for deletes
                return [
                    delta.new_file.path or delta.old_file.path for delta in diff.deltas
                ]
            except Exception:
                return []

        def _safe_str(accessor, default=None):
            """Safely access string attributes that may have encoding issues."""
            try:
                value = accessor()
                return value.strip() if isinstance(value, str) and value else value
            except (LookupError, UnicodeDecodeError):
                # LookupError covers "unknown encoding: System" and similar
                return default

        # Walk all commits
        walker = repo.walk(start_id, pygit2.GIT_SORT_TOPOLOGICAL | pygit2.GIT_SORT_TIME)  # type: ignore[arg-type]
        since_timestamp = since.timestamp()

        # Build a mapping from commit id to tag names
        commit_tags: dict[str, list[str]] = {}
        for ref_name in repo.references:
            if ref_name.startswith("refs/tags/"):
                tag_name = ref_name[len("refs/tags/") :]
                try:
                    ref = repo.references[ref_name]
                    obj = ref.peel()
                    if isinstance(obj, pygit2.Commit):
                        commit_id = str(obj.id)
                        commit_tags.setdefault(commit_id, []).append(tag_name)
                except Exception:
                    continue

        commits_data = []
        for commit in walker:
            # Skip commits older than the cutoff date
            if commit.commit_time < since_timestamp:
                continue
            row = {
                "hash": str(commit.id),
                "author": _safe_str(
                    lambda c=commit: c.author.name if c.author else None
                ),
                "email": _safe_str(
                    lambda c=commit: c.author.email if c.author else None
                ),
                "message": _safe_str(lambda c=commit: c.message),
                "timestamp": commit.commit_time,
                "date": datetime.fromtimestamp(commit.commit_time),
            }
            if include_files:
                row["files"] = _changed_paths(commit)
            # Add tags associated with this commit
            row["tags"] = commit_tags.get(str(commit.id), [])
            commits_data.append(row)

        # Sort by timestamp (oldest to newest)
        commits_data.sort(key=lambda x: x["timestamp"])

        return pd.DataFrame(commits_data)

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve commits from {repo_path}: {e}") from e


def find_upstream_version_tag_commit(
    commits: pd.DataFrame,
    version: str,
) -> str | None:
    """Find the commit hash corresponding to the upstream version tag.

    Args:
        commits: DataFrame of commits with a 'tags' column.
        version: The version string to search for (e.g., '1.2.3').

    Returns:
        The commit hash of the matching tag, or None if not found.

    """
    version_tag_patterns = [
        re.compile(rf"^v{re.escape(version)}$"),  # v1.2.3
        re.compile(rf"^{re.escape(version)}$"),  # 1.2.3
        re.compile(
            rf"^release[-_]?{re.escape(version)}$"
        ),  # release-1.2.3 or release_1.2.3
        re.compile(
            rf"^version[-_]?{re.escape(version)}$"
        ),  # version-1.2.3 or version_1.2.3
    ]

    for _, row in commits.iterrows():
        tags = row.get("tags", [])
        if not isinstance(tags, list):
            continue
        for tag in tags:
            for pattern in version_tag_patterns:
                if pattern.match(tag):
                    return str(row["hash"])
    return None
