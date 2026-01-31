import pytest
import pandas as pd
from osslag.distro.debian import (
    merge_release_packages,
    extract_upstream_version,
    add_upstream_version_column,
)


# Tests for extract_upstream_version function


def test_version_with_debian_revision():
    """Test version string with debian revision."""
    assert extract_upstream_version("1.2.3-4") == "1.2.3"
    assert extract_upstream_version("2.0.1-1") == "2.0.1"
    assert extract_upstream_version("0.5.0-2ubuntu1") == "0.5.0"


def test_version_with_epoch_and_debian_revision():
    """Test version string with epoch and debian revision."""
    assert extract_upstream_version("2:1.2.3-4") == "1.2.3"
    assert extract_upstream_version("1:0.9.0-5") == "0.9.0"
    assert extract_upstream_version("5:4.3.2-1") == "4.3.2"


def test_version_with_epoch_only():
    """Test version string with epoch but no debian revision."""
    assert extract_upstream_version("1:2.0") == "2.0"
    assert extract_upstream_version("3:1.5.0") == "1.5.0"


def test_version_without_epoch_or_revision():
    """Test simple upstream version only."""
    assert extract_upstream_version("1.2.3") == "1.2.3"
    assert extract_upstream_version("0.1.0") == "0.1.0"


def test_version_with_hyphens_in_upstream():
    """Test upstream version that contains hyphens."""
    # Only the LAST hyphen separates debian revision
    assert extract_upstream_version("1.2-alpha-3") == "1.2-alpha"
    assert extract_upstream_version("2.0-rc1-1") == "2.0-rc1"
    assert extract_upstream_version("1:3.0-beta-2") == "3.0-beta"


def test_version_with_complex_formats():
    """Test complex real-world version strings."""
    assert extract_upstream_version("20200101-1") == "20200101"
    assert extract_upstream_version("2:7.4.052-1ubuntu3.1") == "7.4.052"
    assert extract_upstream_version("1.0~rc1-2") == "1.0"


def test_version_with_debian_specific_suffixes():
    """Test version strings with Debian-specific suffixes."""
    assert extract_upstream_version("1.0+dfsg-1") == "1.0"
    assert extract_upstream_version("2.0+ds-2") == "2.0"


def test_version_with_weird_suffix():
    """Test version strings with git suffixes."""
    assert extract_upstream_version("1.10.0+git20230112.1.a437a45-1") == "1.10.0"
    assert extract_upstream_version("1.7+git20191031.4036a9c") == "1.7"
    assert extract_upstream_version("8.8.1+ds+~cs25.17.7") == "8.8.1"


def test_invalid_inputs():
    """Test invalid input handling."""
    assert extract_upstream_version("") is None
    assert extract_upstream_version(None) is None  # type: ignore
    assert extract_upstream_version("   ") is None


def test_whitespace_handling():
    """Test that leading/trailing whitespace is handled."""
    assert extract_upstream_version("  1.2.3-4  ") == "1.2.3"
    assert extract_upstream_version(" 2:1.0-1 ") == "1.0"


# Tests for merge_release_packages function


def test_merge_all_matching():
    """Test merging when all rows match on source column."""
    df1 = pd.DataFrame(
        {
            "source": ["pkg1", "pkg2", "pkg3"],
            "bookworm_version": ["1.0", "2.0", "3.0"],
            "homepage": ["http://a.com", "http://b.com", "http://c.com"],
        }
    )

    df2 = pd.DataFrame(
        {
            "source": ["pkg1", "pkg2", "pkg3"],
            "trixie_version": ["1.1", "2.1", "3.1"],
            "homepage": ["http://a.com", "http://b.com", "http://c.com"],
        }
    )

    matched, unmatched = merge_release_packages([df1, df2])

    # All rows should match
    assert len(matched) == 3
    assert len(unmatched) == 0

    # Check that source column is present
    assert "source" in matched.columns

    # Check that version columns from both releases are present
    assert "bookworm_version" in matched.columns
    assert "trixie_version" in matched.columns


def test_merge_partial_matching():
    """Test merging when only some rows match on source column."""
    df1 = pd.DataFrame(
        {
            "source": ["pkg1", "pkg2", "pkg3"],
            "bookworm_version": ["1.0", "2.0", "3.0"],
            "homepage": ["http://a.com", "http://b.com", "http://c.com"],
        }
    )

    df2 = pd.DataFrame(
        {
            "source": ["pkg2", "pkg3", "pkg4"],
            "trixie_version": ["2.1", "3.1", "4.1"],
            "homepage": ["http://b.com", "http://c.com", "http://d.com"],
        }
    )

    matched, unmatched = merge_release_packages([df1, df2])

    # pkg2 and pkg3 should match
    assert len(matched) == 2
    assert sorted(matched["source"].tolist()) == ["pkg2", "pkg3"]

    # pkg1 and pkg4 should be unmatched
    assert len(unmatched) == 2
    assert "pkg1" in unmatched["source"].tolist()
    assert "pkg4" in unmatched["source"].tolist()


def test_merge_no_matching():
    """Test merging when no rows match on source column."""
    df1 = pd.DataFrame(
        {
            "source": ["pkg1", "pkg2"],
            "bookworm_version": ["1.0", "2.0"],
            "homepage": ["http://a.com", "http://b.com"],
        }
    )

    df2 = pd.DataFrame(
        {
            "source": ["pkg3", "pkg4"],
            "trixie_version": ["3.1", "4.1"],
            "homepage": ["http://c.com", "http://d.com"],
        }
    )

    matched, unmatched = merge_release_packages([df1, df2])

    # No rows should match
    assert len(matched) == 0

    # All rows should be unmatched
    assert len(unmatched) == 4


def test_merge_wrong_number_of_dataframes():
    """Test that function raises error when not given exactly 2 DataFrames."""
    df1 = pd.DataFrame({"source": ["pkg1"], "version": ["1.0"]})
    df2 = pd.DataFrame({"source": ["pkg1"], "version": ["1.1"]})
    df3 = pd.DataFrame({"source": ["pkg1"], "version": ["1.2"]})

    with pytest.raises(ValueError):
        merge_release_packages([df1, df2, df3])

    with pytest.raises(ValueError):
        merge_release_packages([df1])


# Tests for add_upstream_version_column function


def test_add_upstream_version_default_column_name():
    """Test adding upstream version column with default naming."""
    df = pd.DataFrame(
        {
            "source": ["pkg1", "pkg2", "pkg3"],
            "version": ["1.2.3-4", "2:1.0-1", "0.5.0-2ubuntu1"],
        }
    )

    result = add_upstream_version_column(df, "version")

    # Check that new column was added
    assert "version_upstream" in result.columns

    # Check values
    expected_upstream = ["1.2.3", "1.0", "0.5.0"]
    assert result["version_upstream"].tolist() == expected_upstream

    # Check original data is preserved
    assert result["source"].tolist() == ["pkg1", "pkg2", "pkg3"]
    assert result["version"].tolist() == ["1.2.3-4", "2:1.0-1", "0.5.0-2ubuntu1"]


def test_add_upstream_version_custom_column_name():
    """Test adding upstream version column with custom naming."""
    df = pd.DataFrame(
        {"source": ["pkg1", "pkg2"], "package_version": ["1.0-1", "2.0-1"]}
    )

    result = add_upstream_version_column(df, "package_version", "upstream")

    # Check that custom column name was used
    assert "upstream" in result.columns
    assert result["upstream"].tolist() == ["1.0", "2.0"]


def test_add_upstream_version_multiple_releases():
    """Test adding upstream version for multiple release columns."""
    df = pd.DataFrame(
        {
            "source": ["pkg1", "pkg2"],
            "bookworm_version": ["1.0-1", "2.0-2"],
            "trixie_version": ["1.1-1", "2.1-1"],
        }
    )

    # Add upstream version for bookworm
    result = add_upstream_version_column(df, "bookworm_version")
    result = add_upstream_version_column(result, "trixie_version")

    # Check both upstream columns exist
    assert "bookworm_version_upstream" in result.columns
    assert "trixie_version_upstream" in result.columns

    # Check values
    assert result["bookworm_version_upstream"].tolist() == ["1.0", "2.0"]
    assert result["trixie_version_upstream"].tolist() == ["1.1", "2.1"]


def test_add_upstream_version_column_not_found():
    """Test error when version column doesn't exist."""
    df = pd.DataFrame({"source": ["pkg1"], "version": ["1.0-1"]})

    with pytest.raises(ValueError):
        add_upstream_version_column(df, "nonexistent_column")


def test_add_upstream_version_doesnt_modify_original():
    """Test that original DataFrame is not modified."""
    df = pd.DataFrame({"source": ["pkg1", "pkg2"], "version": ["1.2.3-4", "1.0-1"]})

    original_columns = set(df.columns)
    result = add_upstream_version_column(df, "version")

    # Check original DataFrame wasn't modified
    assert set(df.columns) == original_columns
    assert "version_upstream" not in df.columns

    # Check result has new column
    assert "version_upstream" in result.columns
