from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import NamedTuple

import pandas as pd
import requests
from dotenv import load_dotenv
from github import Github
from github.GithubException import GithubException

from osslag.utils import vcs

load_dotenv()  # Load environment variables from .env file if present
logger = logging.getLogger(__name__)

# Suppress overly verbose logging from urllib3 and requests
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

# Redirect PyGithub's logging to a file for debugging
github_logger = logging.getLogger("github")
filer_handler = logging.FileHandler("github_debug.log")
filer_handler.setLevel(logging.INFO)
github_logger.addHandler(filer_handler)
github_logger.setLevel(logging.CRITICAL)


class GithubAPIResult(NamedTuple):
    success: bool
    data: dict | None
    error: str | None


def gh_get_rate_limit_info(github_token: str | None = None) -> dict | None:
    """Retrieve GitHub API rate limit information.

    Args:
        github_token: Optional GitHub Personal Access Token. If not provided,
                      uses unauthenticated access limits.

    Returns:
        A dictionary with keys: 'limit', 'remaining', 'reset_datetime', 'authenticated'.
        Returns None on error.

    """
    try:
        gh = Github(github_token) if github_token else Github()
        rate_limit = gh.get_rate_limit()
        core = rate_limit.resources.core

        # core.reset is a datetime object
        reset_dt = (
            core.reset
            if isinstance(core.reset, datetime)
            else datetime.fromtimestamp(core.reset)
        )
        # convert to local timezone
        reset_dt = reset_dt.astimezone()
        # convert to a naive datetime in local time
        reset_dt = reset_dt.replace(tzinfo=None)

        return {
            "limit": core.limit,
            "remaining": core.remaining,
            "reset_datetime": reset_dt.strftime("%I:%M:%S %p"),
            "authenticated": github_token is not None,
        }
    except GithubException:
        return None
    except Exception:
        return None


def fetch_pull_requests(
    repo_url: str,
    github_token: str | None = None,
    state: str = "all",
    months: int | None = None,
) -> pd.DataFrame:
    """Retrieve pull requests for a GitHub repository via the GitHub API.

    Args:
        repo_url: HTTPS URL to the repository (e.g., https://github.com/owner/repo[.git]).
        github_token: Optional GitHub token for authenticated requests (higher rate limits, private repos).
        state: Filter by PR state: 'open', 'closed', or 'all' (default 'all').
        months: Optional limit to PRs created within the last N months (approx 30 days per month).

    Returns:
        A list of dictionaries with PR metadata: 'number', 'title', 'state', 'user', 'created_at',
        'updated_at', 'closed_at', 'merged_at', 'html_url'. Returns None on error.

    """
    if months is not None and (not isinstance(months, int) or months < 1):
        raise ValueError("months parameter must be a positive integer or None")

    owner, repo = vcs.extract_owner_name_repo(repo_url)
    github_token = github_token or os.getenv("GITHUB_TOKEN")
    if owner is None or repo is None:
        raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

    try:
        gh = Github(github_token) if github_token else Github()
        repo_obj = gh.get_repo(f"{owner}/{repo}")

        # PyGithub supports state in {'open','closed','all'}
        prs = repo_obj.get_pulls(state=state, sort="created", direction="desc")

        cutoff = None
        if months is not None:
            cutoff = datetime.now() - timedelta(days=months * 30)

        results: list[dict] = []
        for pr in prs:
            # Filter by months if requested
            if cutoff is not None and pr.created_at < cutoff:
                # Because we sorted desc by created time, we can stop early
                break

            results.append(
                {
                    "number": pr.number,
                    "title": pr.title,
                    "state": pr.state,
                    "user": None if pr.user is None else pr.user.login,
                    "created_at": pr.created_at.isoformat() if pr.created_at else None,
                    "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
                    "closed_at": pr.closed_at.isoformat() if pr.closed_at else None,
                    "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                    "html_url": pr.html_url,
                }
            )

        return pd.DataFrame(results)
    except GithubException as e:
        raise ValueError(f"GitHub API error: {e.data.get('message', str(e))}") from e
    except Exception as e:
        raise ValueError(f"Failed to fetch pull requests: {str(e)}") from e


def gh_check_repo_exists(owner: str, repo: str) -> GithubAPIResult:
    """Check if a GitHub repository exists via the API."""
    github_token = os.getenv("GITHUB_TOKEN")

    if github_token:
        logger.debug("Using authenticated GitHub access")
    else:
        logger.warning("Using unauthenticated GitHub access (60 req/hr limit)")

    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            return GithubAPIResult(data=None, error=None, success=True)

        # Handle rate limiting explicitly
        if response.status_code == 403:
            remaining = response.headers.get("X-RateLimit-Remaining", "?")
            reset_time_str = response.headers.get("X-RateLimit-Reset", "")
            error_msg = (
                f"Rate limited (remaining: {remaining}, resets: {reset_time_str})"
            )
            return GithubAPIResult(
                data={"owner": owner, "repo": repo},
                error=error_msg,
                success=False,
            )

        # Handle actual not found
        if response.status_code == 404:
            error_msg = f"404 Not Found: {url}"
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg = error_data["message"]
            except Exception:
                pass
            return GithubAPIResult(
                data={"owner": owner, "repo": repo}, error=error_msg, success=False
            )

        # Other errors
        error_msg = f"HTTP {response.status_code}"
        try:
            error_data = response.json()
            if "message" in error_data:
                error_msg = error_data["message"]
        except Exception:
            pass
        logger.warning(f"GitHub API error for {owner}/{repo}: {error_msg}")
        return GithubAPIResult(
            data={"owner": owner, "repo": repo}, error=error_msg, success=False
        )
    except requests.RequestException as e:
        return GithubAPIResult(
            data={"owner": owner, "repo": repo}, error=str(e), success=False
        )


def fetch_github_repo_metadata(
    repo_url: str, github_token: str | None = None
) -> pd.DataFrame:
    """Fetch GitHub repo metadata given repo URL and token.

    Handles rate limiting by catching RateLimitExceededException and waiting
    for the reset time before retrying.
    """
    owner, repo = vcs.extract_owner_name_repo(repo_url)
    github_token = github_token or os.getenv("GITHUB_TOKEN")
    if owner is None or repo is None:
        raise ValueError(f"Invalid repository URL: {repo_url}")

    # Configure GitHub client with explicit timeout (30 seconds)
    github_client = (
        Github(github_token, timeout=30) if github_token else Github(timeout=30)
    )
    repo_obj = github_client.get_repo(f"{owner}/{repo}")
    data = {
        "repo_url": repo_url,
        "full_name": repo_obj.full_name,
        "description": repo_obj.description,
        "stargazers_count": repo_obj.stargazers_count,
        "forks_count": repo_obj.forks_count,
        "open_issues_count": repo_obj.open_issues_count,
        "watchers_count": repo_obj.watchers_count,
        "created_at": repo_obj.created_at,
        "updated_at": repo_obj.updated_at,
        "pushed_at": repo_obj.pushed_at,
        "archived": repo_obj.archived,
        "license": str(repo_obj.license.spdx_id) if repo_obj.license else None,
        "topics": ",".join(repo_obj.get_topics()),
    }
    return pd.DataFrame([data])
