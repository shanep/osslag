"""Semantic Version Activity Categorizer (PVAC)

Author:
    Shane Panter and Luke Hindman

Description:
    This module provides a set of functions for categorizeing version strings
    based on the official and extended semantic versioning policies. The module
    also provides a function for calculating the version delta between two
    packages based on the weighted sum of the major, minor, and patch version
    numbers.
"""

import re
from collections.abc import Sequence
from typing import NamedTuple, TypedDict

"""Regular expression patterns for matching version strings"""


class VersionMapping(TypedDict):
    pattern: re.Pattern[str]
    class_group: str


class VersionInfo(TypedDict):
    category: str | None
    epoch: int | None
    major: int | None
    minor: int | None
    patch: int | None


class VersionDeltaWeights(NamedTuple):
    """
    Constant weights to apply to the major, minor, and patch version numbers
    when calculating the version delta. Constants default to the values used in the original PVAC implementation.
    """

    major: float = 0.3337
    minor: float = 0.3306
    patch: float = 0.3321


class VersionTuple(NamedTuple):
    semantic: str
    epoch: int
    major: int
    minor: int
    patch: int


version_mapping: list[VersionMapping] = [
    # Official Semantic
    {
        "pattern": re.compile(
            r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        ),
        "class_group": "Semantic",
    },
    # ExtendedSemantic: Match epoch prepended to version string based upon official versioning policy
    {
        "pattern": re.compile(
            r"^((?P<epoch>0|[1-9]\d*):)?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        ),
        "class_group": "Extended-Semantic",
    },
    # Semi-Semantic: Allow version numbers to start with 0; Make the patch field optional and separated by either a . or a lower case p
    {
        "pattern": re.compile(
            r"^((?P<epoch>0|[1-9]\d*):)?(?P<major>[0-9]\d*)\.(?P<minor>[0-9]\d*)((\.|p|pl)(?P<patch>[0-9]\d*))?(?:-(?P<prerelease>(?:[0-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:[0-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
        ),
        "class_group": "Semi-Semantic",
    },
]


def lookup_category(version_string: str) -> VersionInfo:
    """Given a version string, return a dictionary containing the category

    Args:
        version_string: The version string to categorize

    Returns:
        A dictionary containing the category information

    """
    # Clean the string for standardized processing
    version_string = version_string.strip()

    version_dict: VersionInfo = {
        "category": None,
        "epoch": None,
        "major": None,
        "minor": None,
        "patch": None,
    }
    for map in version_mapping:
        m = map["pattern"].match(version_string.strip())
        if m is not None:
            groups = m.groupdict()
            version_dict = VersionInfo(
                category=map["class_group"],
                epoch=int(groups["epoch"]) if groups.get("epoch") is not None else 0,
                major=int(groups["major"]) if groups.get("major") is not None else 0,
                minor=int(groups["minor"]) if groups.get("minor") is not None else 0,
                patch=int(groups["patch"]) if groups.get("patch") is not None else 0,
            )
            break
    return version_dict


def version_delta_agg(
    packages: Sequence[tuple[VersionTuple, VersionTuple]],
    weights: VersionDeltaWeights,
) -> float:
    """Calculate the sum of version deltas across multiple package pairs.

    Args:
        packages: A list of tuples containing version information
        weights: The weights to apply to the major, minor, and patch version numbers

    Returns:
        The sum of the weighted version deltas

    """
    return sum(version_delta(a, b, weights) for a, b in packages)


def version_delta(
    a: VersionTuple,
    b: VersionTuple,
    weights: VersionDeltaWeights,
) -> float:
    """Calculate the weighted version delta between two versions.

    Args:
        a: Upstream version tuple (must be >= distro version)
        b: Distro version tuple
        weights: The weights to apply to the major, minor, and patch version numbers

    Returns:
        The weighted version delta. Returns 0.0 if epochs differ or
        either version is Unknown.

    """
    if a.epoch != b.epoch:
        return 0.0

    if a.semantic == "Unknown" or b.semantic == "Unknown":
        return 0.0

    weighted_a = (a.major * weights.major) + (a.minor * weights.minor) + (a.patch * weights.patch)
    weighted_b = (b.major * weights.major) + (b.minor * weights.minor) + (b.patch * weights.patch)
    raw_delta = weighted_a - weighted_b

    if raw_delta < 0:
        raise ValueError(
            f"Negative VND: upstream version {a} must be >= distro version {b} "
            f"(weighted: {weighted_a} < {weighted_b})"
        )

    return raw_delta


def categorize_development_activity(version_string_A: str, version_string_B: str) -> str:
    """Calculate the development activity level between two version strings

    Args:
        version_string_A: The first version string to compare
        version_string_B: The second version string to compare

    Returns:
        A string representing the development activity level between
        the two version strings

    """
    class_A_dict = lookup_category(version_string_A)
    class_B_dict = lookup_category(version_string_B)

    if (
        class_A_dict["category"] == "Unknown"
        or class_A_dict["category"] is None
        or class_B_dict["category"] == "Unknown"
        or class_B_dict["category"] is None
    ):
        return "Unknown"

    if class_A_dict["epoch"] != class_B_dict["epoch"]:
        return "Unknown"

    activity_level = "Unknown"

    if class_A_dict["major"] != class_B_dict["major"]:
        activity_level = "Very Active"
    elif class_A_dict["minor"] != class_B_dict["minor"]:
        activity_level = "Moderately Active"
    elif class_A_dict["patch"] != class_B_dict["patch"]:
        activity_level = "Lightly Active"
    else:
        activity_level = "Sedentary"

    return activity_level
