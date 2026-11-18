//! Debian version strings.

use std::sync::LazyLock;

use regex::Regex;

/// `debian_support.Version`'s grammar: `[epoch:]upstream_version[-debian_revision]`.
static DEBIAN_VERSION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^((?P<epoch>\d+):)?(?P<upstream_version>[A-Za-z0-9.+:~-]+?)(-(?P<debian_revision>[A-Za-z0-9+.~]+))?$")
        .expect("debian version regex")
});

/// Upstream part of a Debian version, without repack (`+dfsg`, `+git...`) or prerelease (`~rc1`)
/// suffixes. Returns `None` for empty or invalid versions.
///
/// `2:7.4.052-1ubuntu3.1` gives `7.4.052`, `1.0~rc1-2` gives `1.0`, `8.8.1+ds+~cs25.17.7` gives `8.8.1`.
pub fn extract_upstream_version(version: &str) -> Option<String> {
    if version.is_empty() {
        return None;
    }
    let caps = DEBIAN_VERSION.captures(version.trim())?;
    let upstream = caps.name("upstream_version")?.as_str();
    if caps.name("epoch").is_none() && upstream.contains(':') {
        return None;
    }
    let upstream = upstream.split('+').next().unwrap_or_default();
    let upstream = upstream.split('~').next().unwrap_or_default().trim();
    (!upstream.is_empty()).then(|| upstream.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_versions() {
        let cases = [
            ("1.2.3-4", Some("1.2.3")),
            ("2:1.2.3-4", Some("1.2.3")),
            ("1:2.0", Some("2.0")),
            ("1.2-alpha-3", Some("1.2-alpha")),
            ("2:7.4.052-1ubuntu3.1", Some("7.4.052")),
            ("1.0~rc1-2", Some("1.0")),
            ("1.10.0+git20230101-1", Some("1.10.0")),
            ("8.8.1+ds+~cs25.17.7", Some("8.8.1")),
            ("", None),
            ("   ", None),
            ("1:2:3", Some("2:3")),
            ("a:b", None),
            ("+dfsg", None),
        ];
        for (input, expected) in cases {
            assert_eq!(extract_upstream_version(input).as_deref(), expected, "{input}");
        }
    }
}
