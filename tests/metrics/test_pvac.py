import pytest

from osslag.metrics.pvac import (
    VersionDeltaWeights,
    VersionTuple,
    lookup_category,
    version_delta_agg,
    categorize_development_activity,
)


# ── lookup_category ──────────────────────────────────────────────────────


class TestLookupCategory:
    """Tests for lookup_category()."""

    # --- Semantic versions ---

    def test_standard_semver(self):
        result = lookup_category("1.2.3")
        assert result["category"] == "Semantic"
        assert result["epoch"] == 0
        assert result["major"] == 1
        assert result["minor"] == 2
        assert result["patch"] == 3

    def test_semver_zeros(self):
        result = lookup_category("0.0.0")
        assert result["category"] == "Semantic"
        assert result["major"] == 0
        assert result["minor"] == 0
        assert result["patch"] == 0

    def test_semver_with_prerelease(self):
        result = lookup_category("1.0.0-alpha.1")
        assert result["category"] == "Semantic"
        assert result["major"] == 1
        assert result["minor"] == 0
        assert result["patch"] == 0

    def test_semver_with_build_metadata(self):
        result = lookup_category("1.0.0+build.123")
        assert result["category"] == "Semantic"
        assert result["major"] == 1

    def test_semver_with_prerelease_and_build(self):
        result = lookup_category("2.1.0-beta.2+build.456")
        assert result["category"] == "Semantic"
        assert result["major"] == 2
        assert result["minor"] == 1
        assert result["patch"] == 0

    def test_semver_large_numbers(self):
        result = lookup_category("100.200.300")
        assert result["category"] == "Semantic"
        assert result["major"] == 100
        assert result["minor"] == 200
        assert result["patch"] == 300

    # --- Extended-Semantic versions (with epoch) ---

    def test_extended_semver_with_epoch(self):
        result = lookup_category("2:1.2.3")
        assert result["category"] == "Extended-Semantic"
        assert result["epoch"] == 2
        assert result["major"] == 1
        assert result["minor"] == 2
        assert result["patch"] == 3

    def test_extended_semver_epoch_zero(self):
        # epoch 0 still matches Extended-Semantic because it has the "0:" prefix
        result = lookup_category("0:1.2.3")
        assert result["category"] == "Extended-Semantic"
        assert result["epoch"] == 0

    # --- Semi-Semantic versions ---

    def test_semi_semver_two_components(self):
        result = lookup_category("1.2")
        assert result["category"] == "Semi-Semantic"
        assert result["major"] == 1
        assert result["minor"] == 2
        assert result["patch"] == 0

    def test_semi_semver_leading_zero(self):
        result = lookup_category("01.02.03")
        assert result["category"] == "Semi-Semantic"
        assert result["major"] == 1
        assert result["minor"] == 2
        assert result["patch"] == 3

    def test_semi_semver_p_separator(self):
        result = lookup_category("1.2p3")
        assert result["category"] == "Semi-Semantic"
        assert result["major"] == 1
        assert result["minor"] == 2
        assert result["patch"] == 3

    def test_semi_semver_pl_separator(self):
        result = lookup_category("1.2pl3")
        assert result["category"] == "Semi-Semantic"
        assert result["major"] == 1
        assert result["minor"] == 2
        assert result["patch"] == 3

    def test_semi_semver_with_epoch(self):
        result = lookup_category("1:3.4")
        assert result["category"] == "Semi-Semantic"
        assert result["epoch"] == 1
        assert result["major"] == 3
        assert result["minor"] == 4
        assert result["patch"] == 0

    # --- Unrecognized versions ---

    def test_unknown_single_number(self):
        result = lookup_category("42")
        assert result["category"] is None
        assert result["major"] is None

    def test_unknown_empty_string(self):
        result = lookup_category("")
        assert result["category"] is None

    def test_unknown_garbage(self):
        result = lookup_category("not-a-version")
        assert result["category"] is None

    # --- Whitespace handling ---

    def test_strips_whitespace(self):
        result = lookup_category("  1.2.3  ")
        assert result["category"] == "Semantic"
        assert result["major"] == 1

    # --- Priority: Semantic matched before Extended/Semi ---

    def test_semantic_takes_priority(self):
        """A standard semver string should match Semantic, not Extended or Semi."""
        result = lookup_category("3.4.5")
        assert result["category"] == "Semantic"


# ── version_delta ────────────────────────────────────────────────────────


V = VersionTuple
W = VersionDeltaWeights


class TestVersionDelta:
    """Tests for version_delta_agg()."""

    def test_single_pair_major_diff(self):
        packages = [
            (V("Semantic", 0, 2, 0, 0), V("Semantic", 0, 1, 0, 0)),
        ]
        # delta = 2*1.0 - 1*1.0 = 1.0
        result = version_delta_agg(packages, W(1.0, 0.1, 0.01))
        assert result == pytest.approx(1.0)

    def test_single_pair_minor_diff(self):
        packages = [
            (V("Semantic", 0, 1, 5, 0), V("Semantic", 0, 1, 3, 0)),
        ]
        # delta = (1 + 0.5) - (1 + 0.3) = 0.2
        result = version_delta_agg(packages, W(1.0, 0.1, 0.01))
        assert result == pytest.approx(0.2)

    def test_single_pair_patch_diff(self):
        packages = [
            (V("Semantic", 0, 1, 0, 10), V("Semantic", 0, 1, 0, 5)),
        ]
        # delta = (1 + 0 + 0.10) - (1 + 0 + 0.05) = 0.05
        result = version_delta_agg(packages, W(1.0, 0.1, 0.01))
        assert result == pytest.approx(0.05)

    def test_identical_versions(self):
        packages = [
            (V("Semantic", 0, 1, 2, 3), V("Semantic", 0, 1, 2, 3)),
        ]
        result = version_delta_agg(packages, W(1.0, 0.1, 0.01))
        assert result == pytest.approx(0.0)

    def test_multiple_pairs_sum(self):
        packages = [
            (V("Semantic", 0, 2, 0, 0), V("Semantic", 0, 1, 0, 0)),
            (V("Semantic", 0, 3, 0, 0), V("Semantic", 0, 1, 0, 0)),
        ]
        # pair1 delta = 1.0, pair2 delta = 2.0 -> total = 3.0
        result = version_delta_agg(packages, W(1.0, 0.1, 0.01))
        assert result == pytest.approx(3.0)

    def test_skips_different_epochs(self):
        packages = [
            (V("Semantic", 1, 5, 0, 0), V("Semantic", 2, 1, 0, 0)),
        ]
        result = version_delta_agg(packages, W(1.0, 0.1, 0.01))
        assert result == pytest.approx(0.0)

    def test_skips_unknown_category(self):
        packages = [
            (V("Unknown", 0, 1, 0, 0), V("Semantic", 0, 2, 0, 0)),
        ]
        result = version_delta_agg(packages, W(1.0, 0.1, 0.01))
        assert result == pytest.approx(0.0)

    def test_skips_both_unknown(self):
        packages = [
            (V("Unknown", 0, 1, 0, 0), V("Unknown", 0, 2, 0, 0)),
        ]
        result = version_delta_agg(packages, W(1.0, 0.1, 0.01))
        assert result == pytest.approx(0.0)

    def test_empty_packages(self):
        result = version_delta_agg([], W(1.0, 0.1, 0.01))
        assert result == pytest.approx(0.0)

    def test_mixed_valid_and_skipped(self):
        packages = [
            (V("Semantic", 0, 2, 0, 0), V("Semantic", 0, 1, 0, 0)),  # valid: delta=1.0
            (V("Semantic", 1, 5, 0, 0), V("Semantic", 2, 1, 0, 0)),  # skipped: epoch mismatch
            (V("Unknown", 0, 1, 0, 0), V("Semantic", 0, 3, 0, 0)),  # skipped: unknown
        ]
        result = version_delta_agg(packages, W(1.0, 0.1, 0.01))
        assert result == pytest.approx(1.0)

    def test_equal_weights(self):
        packages = [
            (V("Semantic", 0, 2, 2, 2), V("Semantic", 0, 1, 1, 1)),
        ]
        # delta = (2+2+2) - (1+1+1) = 3
        result = version_delta_agg(packages, W(1.0, 1.0, 1.0))
        assert result == pytest.approx(3.0)


# ── categorize_development_activity ──────────────────────────────────────


class TestCategorizeDevelopmentActivity:
    """Tests for categorize_development_activity()."""

    def test_very_active_major_change(self):
        assert categorize_development_activity("1.0.0", "2.0.0") == "Very Active"

    def test_moderately_active_minor_change(self):
        assert categorize_development_activity("1.0.0", "1.1.0") == "Moderately Active"

    def test_lightly_active_patch_change(self):
        assert categorize_development_activity("1.0.0", "1.0.1") == "Lightly Active"

    def test_sedentary_identical(self):
        assert categorize_development_activity("1.2.3", "1.2.3") == "Sedentary"

    def test_major_takes_priority_over_minor(self):
        """If major differs, result is Very Active even if minor also differs."""
        assert categorize_development_activity("1.5.0", "2.3.0") == "Very Active"

    def test_major_takes_priority_over_patch(self):
        assert categorize_development_activity("1.0.5", "3.0.1") == "Very Active"

    def test_minor_takes_priority_over_patch(self):
        assert categorize_development_activity("1.0.5", "1.1.1") == "Moderately Active"

    def test_unknown_when_first_unrecognized(self):
        assert categorize_development_activity("garbage", "1.0.0") == "Unknown"

    def test_unknown_when_second_unrecognized(self):
        assert categorize_development_activity("1.0.0", "garbage") == "Unknown"

    def test_unknown_when_both_unrecognized(self):
        assert categorize_development_activity("foo", "bar") == "Unknown"

    def test_unknown_when_epochs_differ(self):
        assert categorize_development_activity("1:1.0.0", "2:1.0.0") == "Unknown"

    def test_same_epoch_works(self):
        assert categorize_development_activity("1:1.0.0", "1:2.0.0") == "Very Active"

    def test_cross_category_semver_and_semi(self):
        """A standard semver vs a two-component semi-semver should still compare."""
        assert categorize_development_activity("1.0.0", "2.0") == "Very Active"

    def test_sedentary_two_component(self):
        assert categorize_development_activity("1.2", "1.2") == "Sedentary"

    def test_lightly_active_with_p_separator(self):
        assert categorize_development_activity("1.2p3", "1.2p4") == "Lightly Active"
