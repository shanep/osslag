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



def version_delta(
    packages: list[tuple[VersionTuple, VersionTuple]],
    major_weight: float,
    minor_weight: float,
    patch_weight: float,
) -> float:
    """Calculate the version delta between two packages based on the weighted
        sum of the major, minor, and patch version numbers.

    Args:
        packages: A list of tuples containing version information
        major_weight: The weight to apply to the major version number
        minor_weight: The weight to apply to the minor version number
        patch_weight: The weight to apply to the patch version number

    Returns:
        A single value representing the sum of the weighted version deltas

    """
    version_delta: float = 0

    for version_tuple_A, version_tuple_B in packages:
        # Destructure the version tuples
        semanticA, epochA, majorA, minorA, patchA = version_tuple_A
        semanticB, epochB, majorB, minorB, patchB = version_tuple_B

        if epochA != epochB:
            continue

        if semanticA == "Unknown" or semanticB == "Unknown":
            continue

        weighted_version_A = (majorA * major_weight) + (minorA * minor_weight) + (patchA * patch_weight)
        weighted_version_B = (majorB * major_weight) + (minorB * minor_weight) + (patchB * patch_weight)
        version_delta += abs(weighted_version_B - weighted_version_A)

    return version_delta


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
