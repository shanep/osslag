import pytest
import tempfile
import shutil
import pathlib
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
from osslag.utils.vcs import (
    find_upstream_version_tag_commit,
    normalize_https_repo_url,
    clone_repo,
    extract_owner_name_repo,
    NormalizeRepoResult,
    label_trivial_commits,
    load_commits,
)


# Tests for normalize_https_repo_url function


def test_strip_fragment():
    """Test URL with trailing garbage after .git."""
    url = "https://github.com/node-modules/address#readme"
    expected = NormalizeRepoResult("https://github.com/node-modules/address", None)
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_https_url_with_garbage():
    """Test HTTPS URL with extra parameters."""
    url = "https://github.com/jazzband/django-haystack-redis cache/repos/django-haystack-redis"
    expected = NormalizeRepoResult("https://github.com/jazzband/django-haystack-redis", None)
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_https_git_url():
    """Test standard HTTPS git URLs."""
    url = "https://github.com/owner/repo.git"
    expected = NormalizeRepoResult("https://github.com/owner/repo", None)
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_url_without_git_extension():
    """Test GitHub URL without .git extension is accepted."""
    url = "https://github.com/owner/repo"
    expected = NormalizeRepoResult("https://github.com/owner/repo", None)
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_url_without_git_extension_trailing_slash():
    """Test GitHub URL without .git extension but with trailing slash is accepted."""
    url = "https://github.com/owner/repo/"
    expected = NormalizeRepoResult("https://github.com/owner/repo", None)
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_url_with_whitespace():
    """Test that leading/trailing whitespace is handled."""
    url = "  https://github.com/owner/repo.git  "
    expected = NormalizeRepoResult("https://github.com/owner/repo", None)
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_convert_http_to_https():
    """Test that http:// URLs are converted to https://."""
    url = "http://github.com/owner/repo.git"
    expected = NormalizeRepoResult("https://github.com/owner/repo", None)
    actual = normalize_https_repo_url(url)
    assert expected == actual


# Valid URL github ssh but not https


def test_ssh_git_url():
    """Test SSH git URLs."""
    url = "git@github.com:owner/repo.git"
    expected = NormalizeRepoResult(None, None)
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_ssh_git2_url():
    """Test SSH git URLs."""
    url = "git@github.com:owner/repo"
    expected = NormalizeRepoResult(None, None)
    actual = normalize_https_repo_url(url)
    assert expected == actual


# Invalid protocol


def test_git_protocol_url():
    """Test git:// protocol URLs."""
    url = "git://github.com/owner/repo.git"
    expected = NormalizeRepoResult(None, "URL does not match expected git repository patterns")
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_ftp_git_url():
    """Test FTP git URLs."""
    url = "ftp://example.com/repo.git"
    expected = NormalizeRepoResult(None, "URL does not match expected git repository patterns")
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_ftps_git_url():
    """Test FTPS git URLs."""
    url = "ftps://example.com/repo.git"
    expected = NormalizeRepoResult(None, "URL does not match expected git repository patterns")
    actual = normalize_https_repo_url(url)
    assert expected == actual


# Invalid URL tests


def test_http_git_url():
    """Test HTTP git URLs."""
    url = "http://example.com/repo.git"
    expected = NormalizeRepoResult(None, "URL does not match expected git repository patterns")
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_http_url_without_git():
    """Test regular HTTP URL (not a git repo) returns None."""
    url = "https://example.com/page.html"
    expected = NormalizeRepoResult(None, "URL does not match expected git repository patterns")
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_plain_string():
    """Test plain string without protocol returns None."""
    url = "not-a-url"
    expected = NormalizeRepoResult(None, "URL does not match expected git repository patterns")
    actual = normalize_https_repo_url(url)
    assert expected == actual


def test_url_with_username():
    """Test SSH URL with username."""
    url = "git@gitlab.com:group/subgroup/project.git"
    expected = NormalizeRepoResult(None, "URL does not match expected git repository patterns")
    actual = normalize_https_repo_url(url)
    assert expected == actual


# Invalid input tests


def test_none_input():
    """Test that None input returns None."""
    expected = NormalizeRepoResult(None, "Invalid URL: None or not a string")
    actual = normalize_https_repo_url(None)  # type: ignore[arg-type]
    assert expected == actual


def test_empty_string():
    """Test empty string returns None."""
    url = ""
    expected = NormalizeRepoResult(None, "Invalid URL: None or not a string")
    actual = normalize_https_repo_url(url)
    assert expected == actual


# Tests for extract_owner_name_repo function


def test_https_without_git_suffix():
    owner, repo = extract_owner_name_repo("https://github.com/foo/bar")
    assert owner == "foo"
    assert repo == "bar"


def test_real_data():
    owner, repo = extract_owner_name_repo("https://github.com/snoyberg/tar-conduit")
    assert owner == "snoyberg"
    assert repo == "tar-conduit"


def test_weird_format():
    owner, repo = extract_owner_name_repo("https://github.com/sboudrias/Inquirer.js")
    assert owner == "sboudrias"
    assert repo == "Inquirer.js"


# Tests for clone_repo function


@pytest.fixture
def test_dir():
    """Create temporary directory for testing."""
    test_dir = tempfile.mkdtemp(prefix="test_clone_")
    yield test_dir
    if pathlib.Path(test_dir).exists():
        shutil.rmtree(test_dir)


def test_clone_repo_invalid_url(test_dir):
    """Test cloning with invalid URL returns failure result."""
    result = clone_repo("not-a-url", test_dir)
    assert not result.success
    assert result.error is not None


def test_clone_repo_empty_url(test_dir):
    """Test cloning with empty URL returns failure result."""
    result = clone_repo("", test_dir)
    assert not result.success
    assert result.error is not None


def test_clone_repo_none_url(test_dir):
    """Test cloning with None URL returns failure result."""
    result = clone_repo(None, test_dir)  # type: ignore
    assert not result.success
    assert result.error is not None


@patch("osslag.utils.github_helper.gh_check_repo_exists")
def test_clone_repo_repo_not_found(mock_check_exists, test_dir):
    """Test cloning non-existent repository returns failure result."""
    from osslag.utils.github_helper import GithubAPIResult

    mock_check_exists.return_value = GithubAPIResult(
        success=False,
        data={"owner": "nonexistent", "repo": "repo"},
        error="Repository not found",
    )

    result = clone_repo("https://github.com/nonexistent/repo", test_dir)
    assert not result.success
    assert result.error is not None


@patch("osslag.utils.github_helper.gh_check_repo_exists")
def test_clone_repo_github_exception(mock_check_exists, test_dir):
    """Test handling of GitHub API exception returns failure result."""
    from osslag.utils.github_helper import GithubAPIResult

    mock_check_exists.return_value = GithubAPIResult(
        success=False, data={"owner": "test", "repo": "repo"}, error="GitHub API error"
    )

    result = clone_repo("https://github.com/test/repo", test_dir)
    assert not result.success
    assert result.error is not None


@patch("pygit2.clone_repository")
@patch("osslag.utils.github_helper.gh_check_repo_exists")
def test_clone_repo_pygit2_error(mock_check_exists, mock_clone, test_dir):
    """Test handling of pygit2 clone error returns failure result and cleans up."""
    import pygit2
    from osslag.utils.github_helper import GithubAPIResult

    mock_check_exists.return_value = GithubAPIResult(success=True, data=None, error=None)

    mock_clone.side_effect = pygit2.GitError("Clone failed")

    result = clone_repo("https://github.com/test/repo", test_dir)
    assert not result.success
    assert result.error is not None


def test_clone_repo_already_exists(test_dir):
    """Test cloning when repo already exists returns success without re-cloning."""
    # Create a fake .git directory directly in dest_dir
    # (clone_repo now clones directly into dest_dir, not a subdirectory)
    repo_path = pathlib.Path(test_dir)
    (repo_path / ".git").mkdir()

    result = clone_repo("https://github.com/test/repo", test_dir)
    assert result.success
    assert result.error is None


@patch("pygit2.clone_repository")
@patch("osslag.utils.github_helper.gh_check_repo_exists")
def test_clone_repo_with_branch(mock_check_exists, mock_clone, test_dir):
    """Test cloning specific branch."""
    from osslag.utils.github_helper import GithubAPIResult

    mock_check_exists.return_value = GithubAPIResult(success=True, data=None, error=None)

    mock_repo_obj = MagicMock()
    mock_clone.return_value = mock_repo_obj

    # Mock the branch checkout
    mock_commit = MagicMock()
    mock_commit.id = b"test_id"
    mock_repo_obj.revparse_single.return_value = mock_commit
    mock_repo_obj.remotes = {"origin": MagicMock()}

    result = clone_repo("https://github.com/test/repo", test_dir, branch="develop")

    # Verify branch checkout was attempted
    assert mock_repo_obj.revparse_single.called
    assert result.success


@patch("pygit2.clone_repository")
@patch("osslag.utils.github_helper.gh_check_repo_exists")
def test_clone_repo_with_tags(mock_check_exists, mock_clone, test_dir):
    """Test cloning with include_tags=True fetches tags."""
    from osslag.utils.github_helper import GithubAPIResult

    mock_check_exists.return_value = GithubAPIResult(success=True, data=None, error=None)

    mock_repo_obj = MagicMock()
    mock_clone.return_value = mock_repo_obj

    mock_remote = MagicMock()
    mock_repo_obj.remotes = {"origin": mock_remote}

    result = clone_repo("https://github.com/test/repo", test_dir, include_tags=True)

    # Verify remote.fetch() was called for tags
    assert mock_remote.fetch.called
    assert result.success


@patch("pygit2.clone_repository")
@patch("osslag.utils.github_helper.gh_check_repo_exists")
def test_clone_repo_without_tags(mock_check_exists, mock_clone, test_dir):
    """Test cloning with include_tags=False skips tag fetching."""
    from osslag.utils.github_helper import GithubAPIResult

    mock_check_exists.return_value = GithubAPIResult(success=True, data=None, error=None)

    mock_repo_obj = MagicMock()
    mock_clone.return_value = mock_repo_obj

    mock_remote = MagicMock()
    mock_repo_obj.remotes = {"origin": mock_remote}

    result = clone_repo("https://github.com/test/repo", test_dir, include_tags=False)

    # Verify remote.fetch() was NOT called for tags
    assert not mock_remote.fetch.called
    assert result.success


@patch("os.getenv")
@patch("pygit2.clone_repository")
@patch("osslag.utils.github_helper.gh_check_repo_exists")
def test_clone_repo_with_auth_token(mock_check_exists, mock_clone, mock_getenv, test_dir):
    """Test cloning with GitHub token authentication."""
    from osslag.utils.github_helper import GithubAPIResult

    mock_getenv.return_value = "test_token_123"

    mock_check_exists.return_value = GithubAPIResult(success=True, data=None, error=None)

    mock_repo_obj = MagicMock()
    mock_clone.return_value = mock_repo_obj

    mock_remote = MagicMock()
    mock_repo_obj.remotes = {"origin": mock_remote}

    result = clone_repo("https://github.com/test/repo", test_dir)

    # Verify clone succeeded with token authentication
    assert result.success
    # Verify clone was called (credentials are passed via callbacks, not directly visible)
    assert mock_clone.called


# Tests for label_trivial_commits function


def test_labels_md_files_as_trivial():
    """Test that commits with only .md files are marked as trivial."""
    df = pd.DataFrame(
        [
            {"files": ["docs/README.md"]},  # Only .md file -> trivial
            {"files": ["README.md"]},  # Only .md file -> trivial
            {"files": ["readme.md", "CHANGELOG.md"]},  # All .md files -> trivial
            {"files": ["readme.md", "src/app.py"]},  # Mixed -> not trivial
            {"files": ["src/main.py"]},  # No .md files -> not trivial
            {"files": []},  # Empty file list -> trivial
        ]
    )

    result = label_trivial_commits(df.copy())

    assert "is_trivial" in result.columns
    expected_flags = [True, True, True, False, False, True]
    assert result["is_trivial"].tolist() == expected_flags


def test_missing_files_column_no_label():
    """Test that missing files column doesn't add label."""
    df = pd.DataFrame(
        [
            {"hash": "abc"},
        ]
    )

    result = label_trivial_commits(df.copy())

    assert "is_trivial" not in result.columns
    assert set(result.columns) == set(df.columns)


# Integration tests for load_commits using shanep/demo repository


@pytest.fixture(scope="module")
def demo_repo():
    """Clone demo repository once for all tests."""
    test_cache = pathlib.Path(tempfile.mkdtemp(prefix="test_get_all_commits_"))
    test_repo_url = "https://github.com/shanep/demo"

    result = clone_repo(test_repo_url, test_cache)
    if result.success:
        # clone_repo clones directly into dest_dir
        if (test_cache / ".git").exists():
            repo_path = test_cache
        else:
            repo_path = None
    else:
        repo_path = None

    yield repo_path

    if test_cache.exists():
        shutil.rmtree(test_cache)


def test_get_all_commits_returns_dataframe(demo_repo):
    """Test that get_all_commits returns a valid DataFrame."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    df = load_commits(demo_repo, since=datetime(1970, 1, 1))

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 13


def test_get_all_commits_has_expected_columns(demo_repo):
    """Test that DataFrame has all expected columns."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    df = load_commits(demo_repo, since=datetime(1970, 1, 1))

    assert df is not None
    expected_columns = {
        "hash",
        "author",
        "email",
        "message",
        "timestamp",
        "date",
        "files",
        "tags",
    }
    assert set(df.columns) == expected_columns


def test_get_all_commits_without_files(demo_repo):
    """Test that include_files=False excludes files column."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    df = load_commits(demo_repo, include_files=False, since=datetime(1970, 1, 1))

    assert df is not None
    assert "files" not in df.columns
    expected_columns = {
        "hash",
        "author",
        "email",
        "message",
        "timestamp",
        "date",
        "tags",
    }
    assert set(df.columns) == expected_columns


def test_get_all_commits_with_files(demo_repo):
    """Test that include_files=True includes files column with valid data."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    df = load_commits(demo_repo, include_files=True, since=datetime(1970, 1, 1))

    assert df is not None
    assert "files" in df.columns

    # Every row should have a list in the files column
    for files in df["files"]:
        assert isinstance(files, list)

    # At least some commits should have changed files (non-empty lists)
    non_empty_files = df[df["files"].apply(len) > 0]
    assert len(non_empty_files) > 0

    # Files should be strings (file paths)
    for files in df["files"]:
        for f in files:
            assert isinstance(f, str)


def test_get_all_commits_sorted_by_timestamp(demo_repo):
    """Test that commits are sorted oldest to newest."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    df = load_commits(demo_repo, include_files=False, since=datetime(1970, 1, 1))

    assert df is not None
    timestamps = df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


def test_get_all_commits_valid_data_types(demo_repo):
    """Test that column data types are correct."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    df = load_commits(demo_repo, since=datetime(1970, 1, 1))

    assert df is not None
    # Check first row has valid types
    assert isinstance(df["hash"].iloc[0], str)
    assert len(df["hash"].iloc[0]) == 40
    # timestamp can be int, float, or numpy numeric type
    first_timestamp = df["timestamp"].iloc[0]
    assert isinstance(first_timestamp, (int, float)) or hasattr(first_timestamp, "__int__")
    assert isinstance(df["date"].iloc[0], datetime)
    assert isinstance(df["files"].iloc[0], list)


def test_get_all_commits_first_commit_author(demo_repo):
    """Test that the first commit is from Shane K. Panter."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    df = load_commits(demo_repo, include_files=False, since=datetime(1970, 1, 1))

    assert df is not None
    # shanep/demo was created by Shane K. Panter
    assert "Shane" in df["author"].iloc[0]


def test_get_all_commits_tags_are_lists(demo_repo):
    """Test that tags column contains lists."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    df = load_commits(demo_repo, include_files=False, since=datetime(1970, 1, 1))

    assert df is not None
    assert "tags" in df.columns
    # Every row should have a list in the tags column
    for tags in df["tags"]:
        assert isinstance(tags, list)


def test_get_all_commits_finds_v1_tag(demo_repo):
    """Test that v1.0.0 tag is associated with the correct commit."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    df = load_commits(demo_repo, include_files=False, since=datetime(1970, 1, 1))

    assert df is not None
    # Find commit with v1.0.0 tag (commit hash: 7add680f5ecad3d9cc88eeb456595054135931e2)
    tagged_commits = df[df["tags"].apply(lambda t: "v1.0.0" in t)]
    assert len(tagged_commits) == 1
    assert tagged_commits.iloc[0]["hash"].startswith("7add680f")


def test_get_all_commits_nonexistent_path():
    """Test that nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_commits("/nonexistent/path/to/repo")


def test_get_all_commits_not_a_repo():
    """Test that non-repo directory raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            load_commits(tmpdir)


def test_get_all_commits_since_date_filters(demo_repo):
    """Test that since parameter filters out older commits."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    # Get all commits first
    df_all = load_commits(demo_repo, since=datetime(1970, 1, 1))
    # Filter with recent date should return fewer commits
    df_recent = load_commits(demo_repo, since=datetime(2025, 1, 1))

    assert df_all is not None
    assert df_recent is not None
    # Recent filter should return fewer or equal commits
    assert len(df_recent) <= len(df_all)


def test_get_all_commits_default_since_is_4_years(demo_repo):
    """Test that default since is 4 years from now."""
    if demo_repo is None:
        pytest.skip("Test repository not available")

    # Default behavior (4 years) should filter out very old commits
    df_default = load_commits(demo_repo)
    df_all = load_commits(demo_repo, since=datetime(1970, 1, 1))

    assert df_default is not None
    assert df_all is not None
    # shanep/demo commits are older than 4 years, so default should return fewer
    assert len(df_default) < len(df_all)


# Tests for find_upstream_version_tag_commit function


def test_exact_version_match():
    """Test finding exact version match without prefix."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": ["1.0.0"]},
            {"hash": "def456", "tags": ["1.1.0"]},
            {"hash": "ghi789", "tags": ["2.0.0"]},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "1.1.0")
    assert result == "def456"


def test_version_with_v_prefix():
    """Test finding version with 'v' prefix."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": ["v1.0.0"]},
            {"hash": "def456", "tags": ["v1.1.0"]},
            {"hash": "ghi789", "tags": ["v2.0.0"]},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "1.1.0")
    assert result == "def456"


def test_no_matching_tag():
    """Test when no tag matches the version."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": ["1.0.0"]},
            {"hash": "def456", "tags": ["1.5.0"]},
            {"hash": "ghi789", "tags": ["2.0.0"]},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "3.0.0")
    assert result is None


def test_empty_tag_list():
    """Test when commits have no tags."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": []},
            {"hash": "def456", "tags": []},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "1.0.0")
    assert result is None


def test_empty_dataframe():
    """Test with empty commits DataFrame."""
    commits_df = pd.DataFrame(columns=["hash", "tags"])

    result = find_upstream_version_tag_commit(commits_df, "1.0.0")
    assert result is None


def test_mixed_tag_formats():
    """Test with mixed tag naming conventions."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": ["1.0.0", "old-tag"]},
            {"hash": "def456", "tags": ["v1.1.0"]},
            {"hash": "ghi789", "tags": ["release-3.0.0"]},
        ]
    )

    # Test exact match
    result = find_upstream_version_tag_commit(commits_df, "1.0.0")
    assert result == "abc123"

    # Test v prefix is found
    result = find_upstream_version_tag_commit(commits_df, "1.1.0")
    assert result == "def456"

    # Test release prefix
    result = find_upstream_version_tag_commit(commits_df, "3.0.0")
    assert result == "ghi789"


def test_commit_with_multiple_tags():
    """Test commit with multiple tags, one matching."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": ["latest", "v1.0.0", "stable"]},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "1.0.0")
    assert result == "abc123"


def test_version_underscore_prefix():
    """Test finding version with 'version_' prefix."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": ["version_1.0.0"]},
            {"hash": "def456", "tags": ["version-1.1.0"]},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "1.0.0")
    assert result == "abc123"

    result = find_upstream_version_tag_commit(commits_df, "1.1.0")
    assert result == "def456"


def test_release_underscore_prefix():
    """Test finding version with 'release_' prefix."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": ["release_2.0.0"]},
            {"hash": "def456", "tags": ["release-2.1.0"]},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "2.0.0")
    assert result == "abc123"

    result = find_upstream_version_tag_commit(commits_df, "2.1.0")
    assert result == "def456"


def test_complex_version_formats():
    """Test finding complex version formats with special characters."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": ["1.2.3+dfsg"]},
            {"hash": "def456", "tags": ["v1.2.3+dfsg"]},
            {"hash": "ghi789", "tags": ["2.0.0~rc1"]},
            {"hash": "jkl012", "tags": ["v2.0.0~rc1"]},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "1.2.3+dfsg")
    assert result == "abc123"

    result = find_upstream_version_tag_commit(commits_df, "2.0.0~rc1")
    assert result == "ghi789"


def test_first_match_returned():
    """Test that first matching commit is returned when multiple commits have matching tags."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": ["v1.0.0"]},
            {"hash": "def456", "tags": ["1.0.0"]},  # Also matches but comes later
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "1.0.0")
    # Should return first match (v1.0.0 matches first)
    assert result == "abc123"


def test_missing_tags_column():
    """Test behavior when tags column is missing."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123"},
            {"hash": "def456"},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "1.0.0")
    assert result is None


def test_non_list_tags():
    """Test behavior when tags is not a list."""
    commits_df = pd.DataFrame(
        [
            {"hash": "abc123", "tags": "v1.0.0"},  # String instead of list
            {"hash": "def456", "tags": None},
        ]
    )

    result = find_upstream_version_tag_commit(commits_df, "1.0.0")
    assert result is None
