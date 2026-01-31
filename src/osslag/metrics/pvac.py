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

"""Regular expression patterns for matching version strings"""
version_mapping = [
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


def lookup_category(version_string):
    """Given a version string, return a dictionary containing the category

    Args:
        version_string (string): The version string to categorize

    Returns:
        dict: A dictionary containing the category information

    """
    # Clean the string for standardized processing
    version_string = version_string.strip()

    version_dict = {
        "category": None,
        "epoch": None,
        "major": None,
        "minor": None,
        "patch": None,
    }
    for map in version_mapping:
        m = map["pattern"].match(version_string.strip())
        if m is not None:
            version_dict = {}
            if "epoch" not in m.groupdict().keys() or m.groupdict()["epoch"] is None:
                version_dict["epoch"] = 0
            else:
                version_dict["epoch"] = int(m.groupdict()["epoch"])

            if "major" not in m.groupdict().keys() or m.groupdict()["major"] is None:
                version_dict["major"] = 0
            else:
                version_dict["major"] = int(m.groupdict()["major"])

            if "minor" not in m.groupdict().keys() or m.groupdict()["minor"] is None:
                version_dict["minor"] = 0
            else:
                version_dict["minor"] = int(m.groupdict()["minor"])

            if "patch" not in m.groupdict().keys() or m.groupdict()["patch"] is None:
                version_dict["patch"] = 0
            else:
                version_dict["patch"] = int(m.groupdict()["patch"])

            version_dict["category"] = map["class_group"]
            break
    return version_dict


def version_delta(packages, major_weight, minor_weight, patch_weight):
    """Calculate the version delta between two packages based on the weighted
        sum of the major, minor, and patch version numbers.

    Args:
        packages (tuple): A list of tuples containing version information
        major_weight (float): The weight to apply to the major version number
        minor_weight (float): The weight to apply to the minor version number
        patch_weight (float): The weight to apply to the patch version number

    Returns:
        type: A single value representing the sum of the weighted version deltas

    """
    version_delta = 0

    for version_tuple_A, version_tuple_B in packages:
        # Destructure the version tuples
        semanticA, epochA, majorA, minorA, patchA = version_tuple_A
        semanticB, epochB, majorB, minorB, patchB = version_tuple_B

        if epochA != epochB:
            continue

        if semanticA == "Unknown" or semanticB == "Unknown":
            continue

        weighted_version_A = (
            (majorA * major_weight) + (minorA * minor_weight) + (patchA * patch_weight)
        )
        weighted_version_B = (
            (majorB * major_weight) + (minorB * minor_weight) + (patchB * patch_weight)
        )
        version_delta += abs(weighted_version_B - weighted_version_A)

    return version_delta


def categorize_development_activity(version_string_A, version_string_B):
    """Calculate the development activity level between two version strings

    Args:
        version_string_A (string): The first version string to compare
        version_string_B (string): The second version string to compare

    Returns:
        string: A string representing the development activity level between
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
