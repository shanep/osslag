//! Package Version Activity Categorizer (PVAC).
//!
//! Sorts version strings into the official semantic, extended semantic (with an epoch) and
//! semi-semantic policies, and measures the weighted version delta between two versions.
//! Authors of the original PVAC: Shane Panter and Luke Hindman.

use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Version policy a string matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Category {
    #[serde(rename = "Semantic")]
    Semantic,
    #[serde(rename = "Extended-Semantic")]
    ExtendedSemantic,
    #[serde(rename = "Semi-Semantic")]
    SemiSemantic,
}

impl Category {
    pub fn as_str(self) -> &'static str {
        match self {
            Category::Semantic => "Semantic",
            Category::ExtendedSemantic => "Extended-Semantic",
            Category::SemiSemantic => "Semi-Semantic",
        }
    }
}

/// A categorized version. Missing epoch or patch parts are 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionInfo {
    pub category: Category,
    pub epoch: u64,
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

/// Weights for the major, minor and patch parts in the version delta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VersionDeltaWeights {
    pub major: f64,
    pub minor: f64,
    pub patch: f64,
}

impl Default for VersionDeltaWeights {
    /// The weights from the original PVAC paper.
    fn default() -> Self {
        Self { major: 0.3337, minor: 0.3306, patch: 0.3321 }
    }
}

/// A version with its policy name, used for delta calculations.
/// `semantic` is "Unknown" when the version did not match any policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionTuple {
    pub semantic: String,
    pub epoch: u64,
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

#[derive(Debug, Error, PartialEq)]
#[error("Negative VND: upstream version {a:?} must be >= distro version {b:?} (weighted: {weighted_a} < {weighted_b})")]
pub struct NegativeDeltaError {
    pub a: Box<VersionTuple>,
    pub b: Box<VersionTuple>,
    pub weighted_a: f64,
    pub weighted_b: f64,
}

const PRERELEASE_BUILD: &str = r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$";

static PATTERNS: LazyLock<[(Regex, Category); 3]> = LazyLock::new(|| {
    let semantic = format!(r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*){PRERELEASE_BUILD}");
    let extended = format!(
        r"^((?P<epoch>0|[1-9]\d*):)?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*){PRERELEASE_BUILD}"
    );
    let semi = r"^((?P<epoch>0|[1-9]\d*):)?(?P<major>[0-9]\d*)\.(?P<minor>[0-9]\d*)((\.|p|pl)(?P<patch>[0-9]\d*))?(?:-(?P<prerelease>(?:[0-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:[0-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$";
    [
        (Regex::new(&semantic).expect("semantic regex"), Category::Semantic),
        (Regex::new(&extended).expect("extended regex"), Category::ExtendedSemantic),
        (Regex::new(semi).expect("semi-semantic regex"), Category::SemiSemantic),
    ]
});

/// Parse a run of ASCII digits. Returns `None` on overflow or a non-ASCII digit.
fn parse_digits(s: &str) -> Option<u64> {
    s.chars().try_fold(0u64, |acc, c| acc.checked_mul(10)?.checked_add(u64::from(c.to_digit(10)?)))
}

/// Categorize a version string. Returns `None` when no policy matches.
pub fn lookup_category(version: &str) -> Option<VersionInfo> {
    let version = version.trim();
    for (pattern, category) in PATTERNS.iter() {
        if let Some(caps) = pattern.captures(version) {
            let part = |name: &str| caps.name(name).map_or(Some(0), |m| parse_digits(m.as_str()));
            return Some(VersionInfo {
                category: *category,
                epoch: part("epoch")?,
                major: part("major")?,
                minor: part("minor")?,
                patch: part("patch")?,
            });
        }
    }
    None
}

fn weighted(v: &VersionTuple, w: &VersionDeltaWeights) -> f64 {
    (v.major as f64 * w.major) + (v.minor as f64 * w.minor) + (v.patch as f64 * w.patch)
}

/// Weighted version delta from distro version `b` up to upstream version `a`.
/// Returns 0 when the epochs differ or either version is Unknown.
pub fn version_delta(a: &VersionTuple, b: &VersionTuple, w: &VersionDeltaWeights) -> Result<f64, NegativeDeltaError> {
    if a.epoch != b.epoch || a.semantic == "Unknown" || b.semantic == "Unknown" {
        return Ok(0.0);
    }
    let weighted_a = weighted(a, w);
    let weighted_b = weighted(b, w);
    let raw = weighted_a - weighted_b;
    if raw < 0.0 {
        return Err(NegativeDeltaError { a: Box::new(a.clone()), b: Box::new(b.clone()), weighted_a, weighted_b });
    }
    Ok(raw)
}

/// Sum of version deltas over package pairs, added left to right.
pub fn version_delta_agg(
    packages: &[(VersionTuple, VersionTuple)],
    w: &VersionDeltaWeights,
) -> Result<f64, NegativeDeltaError> {
    let deltas = packages.iter().map(|(a, b)| version_delta(a, b, w)).collect::<Result<Vec<f64>, _>>()?;
    Ok(crate::metrics::pysum::py_sum(deltas))
}

/// Development activity between two version strings: "Very Active", "Moderately Active",
/// "Lightly Active", "Sedentary" or "Unknown".
pub fn categorize_development_activity(a: &str, b: &str) -> &'static str {
    let (Some(a), Some(b)) = (lookup_category(a), lookup_category(b)) else {
        return "Unknown";
    };
    if a.epoch != b.epoch {
        "Unknown"
    } else if a.major != b.major {
        "Very Active"
    } else if a.minor != b.minor {
        "Moderately Active"
    } else if a.patch != b.patch {
        "Lightly Active"
    } else {
        "Sedentary"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vt(semantic: &str, epoch: u64, major: u64, minor: u64, patch: u64) -> VersionTuple {
        VersionTuple { semantic: semantic.into(), epoch, major, minor, patch }
    }

    #[test]
    fn categories() {
        assert_eq!(lookup_category("1.2.3").unwrap().category, Category::Semantic);
        assert_eq!(lookup_category("1:1.2.3").unwrap().category, Category::ExtendedSemantic);
        assert_eq!(lookup_category("0:1.2.3").unwrap().category, Category::ExtendedSemantic);
        let semi = lookup_category("01.2p7").unwrap();
        assert_eq!(semi.category, Category::SemiSemantic);
        assert_eq!((semi.major, semi.minor, semi.patch), (1, 2, 7));
        assert_eq!(lookup_category(" 1.2 ").unwrap().patch, 0);
        assert!(lookup_category("abc").is_none());
        assert!(lookup_category("1.2.3.4").is_none());
    }

    #[test]
    fn activity() {
        assert_eq!(categorize_development_activity("2.0.0", "1.0.0"), "Very Active");
        assert_eq!(categorize_development_activity("1.1.0", "1.0.0"), "Moderately Active");
        assert_eq!(categorize_development_activity("1.0.1", "1.0.0"), "Lightly Active");
        assert_eq!(categorize_development_activity("1.0.0", "1.0.0"), "Sedentary");
        assert_eq!(categorize_development_activity("1:1.0.0", "1.0.0"), "Unknown");
        assert_eq!(categorize_development_activity("x", "1.0.0"), "Unknown");
    }

    #[test]
    fn delta() {
        let w = VersionDeltaWeights::default();
        assert_eq!(version_delta(&vt("Semantic", 0, 2, 0, 0), &vt("Semantic", 0, 1, 0, 0), &w), Ok(0.3337));
        assert_eq!(version_delta(&vt("Semantic", 1, 2, 0, 0), &vt("Semantic", 0, 1, 0, 0), &w), Ok(0.0));
        assert_eq!(version_delta(&vt("Unknown", 0, 9, 0, 0), &vt("Semantic", 0, 1, 0, 0), &w), Ok(0.0));
        assert!(version_delta(&vt("Semantic", 0, 1, 0, 0), &vt("Semantic", 0, 2, 0, 0), &w).is_err());
        let pairs = vec![
            (vt("Semantic", 0, 2, 0, 0), vt("Semantic", 0, 1, 0, 0)),
            (vt("Semantic", 0, 1, 1, 0), vt("Semantic", 0, 1, 0, 0)),
        ];
        let second = (1.0 * 0.3337 + 1.0 * 0.3306 + 0.0 * 0.3321) - (1.0 * 0.3337 + 0.0 * 0.3306 + 0.0 * 0.3321);
        assert_eq!(version_delta_agg(&pairs, &w), Ok(crate::metrics::pysum::py_sum([0.3337, second])));
    }
}
