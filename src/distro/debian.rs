//! Debian source packages: fetch `Sources.xz`, keep the packages hosted on GitHub, and merge
//! releases.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use regex::Regex;
use tracing::{info, warn};

use crate::distro::version::extract_upstream_version;
use crate::io::table::StringTable;
use crate::vcs::url::{construct_repo_local_path, normalize_https_repo_url};

pub const SOURCES_URL_TEMPLATE: &str = "https://ftp.debian.org/debian/dists/{release}/main/source/Sources.xz";

/// Download `Sources.xz` for a release and parse it into one row per source package.
pub fn fetch_packages(release: &str) -> Result<StringTable> {
    let url = SOURCES_URL_TEMPLATE.replace("{release}", release);
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(60)).build()?;
    let response = client.get(&url).send().with_context(|| format!("fetching {url}"))?.error_for_status()?;
    let xz = response.bytes().with_context(|| format!("reading {url}"))?;
    parse_sources(xz2::read::XzDecoder::new(xz.as_ref()), release)
}

/// Parse a decompressed `Sources` file. Columns: `source`, `{release}_version`, `homepage`,
/// `depends`, `maintainer`.
pub fn parse_sources(reader: impl Read, release: &str) -> Result<StringTable> {
    let mut table =
        StringTable::with_columns(&["source", &format!("{release}_version"), "homepage", "depends", "maintainer"]);
    let mut reader = BufReader::new(reader);
    let mut stanza: Vec<u8> = Vec::new();
    let mut line: Vec<u8> = Vec::new();
    let flush = |stanza: &mut Vec<u8>, table: &mut StringTable| {
        if !stanza.trim_ascii().is_empty() {
            table.push_row(stanza_row(stanza));
        }
        stanza.clear();
    };
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line).context("decompressing Sources")? == 0 {
            break;
        }
        if line.trim_ascii().is_empty() {
            flush(&mut stanza, &mut table);
        } else {
            stanza.extend_from_slice(&line);
        }
    }
    flush(&mut stanza, &mut table);
    Ok(table)
}

fn stanza_row(stanza: &[u8]) -> Vec<Option<String>> {
    let fields = parse_deb822(&decode_utf8_ignore(stanza));
    let get = |k: &str| fields.get(k).cloned();
    let source = get("package").filter(|p| !p.is_empty()).or_else(|| get("source"));
    vec![source, get("version"), get("homepage"), get("depends"), get("maintainer")]
}

/// UTF-8 decoding that drops invalid bytes, like Python's `decode("utf-8", "ignore")`.
fn decode_utf8_ignore(bytes: &[u8]) -> String {
    bytes.utf8_chunks().map(|c| c.valid()).collect()
}

/// Python's `str.splitlines()`, which also breaks on `\v`, `\f`, `\x1c`-`\x1e`, `\x85`, U+2028
/// and U+2029.
fn python_splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let breaks =
            matches!(c, '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if breaks {
            lines.push(&text[start..i]);
            start = i + c.len_utf8();
            if c == '\r' && chars.peek().is_some_and(|(_, n)| *n == '\n') {
                chars.next();
                start += 1;
            }
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

static FIELD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<key>[^: \t\n\r\x0c\x0b]+)\s*:\s*(?P<data>(?:\S+(\s+\S+)*)?)\s*$").expect("deb822 field regex")
});

/// One deb822 paragraph as lowercase field names to values, following python-debian's
/// `Deb822`: comment lines are skipped, continuation lines are joined with `\n`, and a repeated
/// field keeps its last value.
pub fn parse_deb822(text: &str) -> HashMap<String, String> {
    let mut fields = HashMap::new();
    let mut key: Option<String> = None;
    let mut content = String::new();
    let mut at_beginning = true;
    for line in python_splitlines(text) {
        if line.starts_with('#') {
            continue;
        }
        if at_beginning {
            if line.is_empty() {
                continue;
            }
            at_beginning = false;
        }
        if let Some(caps) = FIELD.captures(line) {
            if let Some(k) = key.take() {
                fields.insert(k, std::mem::take(&mut content));
            }
            key = Some(caps["key"].to_lowercase());
            content = caps["data"].to_string();
        } else if line.starts_with(char::is_whitespace) && !line.chars().all(char::is_whitespace) {
            content.push('\n');
            content.push_str(line);
        }
    }
    if let Some(k) = key {
        fields.insert(k, content);
    }
    fields
}

static GITHUB: LazyLock<Regex> = LazyLock::new(|| Regex::new("github.com").expect("github regex"));

/// Keep packages whose homepage is a GitHub repo, with the homepage normalized and one row per repo.
pub fn filter_github_repos(packages: &StringTable) -> Result<StringTable> {
    let Some(homepages) = packages.column("homepage") else {
        bail!("DataFrame must contain 'homepage' column to filter GitHub repositories.");
    };
    // pandas' str.contains treats the pattern as a regex, so the dot matches any character.
    let mask: Vec<bool> = homepages.iter().map(|h| h.as_deref().is_some_and(|h| GITHUB.is_match(h))).collect();
    let mut filtered = packages.filter(&mask);
    let normalized: Vec<Option<String>> = filtered
        .column("homepage")
        .unwrap_or_default()
        .iter()
        .map(|h| normalize_https_repo_url(h.as_deref()).url)
        .collect();
    let mut seen = HashSet::new();
    let keep: Vec<bool> = normalized.iter().map(|u| u.as_ref().is_some_and(|u| seen.insert(u.clone()))).collect();
    filtered.set_column("homepage", normalized);
    Ok(filtered.filter(&keep))
}

/// Add `repo_cache_path`: the local clone of each homepage if it already exists.
pub fn add_local_repo_cache_path_column(packages: &mut StringTable, cache_dir: &Path) -> Result<()> {
    let urls = packages.require("homepage", "repo cache path")?;
    let paths = urls
        .iter()
        .map(|u| {
            u.as_deref()
                .and_then(|u| construct_repo_local_path(u, cache_dir, true))
                .map(|p| p.to_string_lossy().into_owned())
        })
        .collect();
    packages.set_column("repo_cache_path", paths);
    Ok(())
}

/// Add a column with the upstream part of each Debian version in `version_column`.
pub fn add_upstream_version_column(packages: &mut StringTable, version_column: &str, new_column: &str) -> Result<()> {
    let Some(versions) = packages.column(version_column) else {
        bail!("Column '{version_column}' not found in DataFrame");
    };
    let upstream = versions.iter().map(|v| v.as_deref().and_then(extract_upstream_version)).collect();
    packages.set_column(new_column, upstream);
    Ok(())
}

/// Problems that make a merge on `key` unsafe, plus warnings that do not.
#[derive(Debug, Default, PartialEq)]
pub struct MergeCheck {
    pub is_safe: bool,
    pub warnings: Vec<String>,
}

pub fn validate_merge_safety(a: &StringTable, b: &StringTable, key: &str) -> MergeCheck {
    let mut check = MergeCheck { is_safe: true, warnings: Vec::new() };
    for (t, which) in [(a, "first"), (b, "second")] {
        if !t.has_column(key) {
            check.warnings.push(format!("Merge key '{key}' missing in {which} DataFrame"));
            check.is_safe = false;
        }
    }
    if !check.is_safe {
        return check;
    }
    let (ka, kb) = (a.column(key).unwrap_or_default(), b.column(key).unwrap_or_default());
    for (keys, which) in [(ka, "First"), (kb, "Second")] {
        let mut seen = HashSet::new();
        let dupes = keys.iter().filter(|k| !seen.insert(*k)).count();
        if dupes > 0 {
            check.warnings.push(format!("{which} DataFrame has {dupes} duplicate '{key}' values"));
        }
    }
    let set_a: HashSet<_> = ka.iter().collect();
    let overlap = kb.iter().collect::<HashSet<_>>().intersection(&set_a).count();
    let pct = overlap as f64 / a.len().max(b.len()).max(1) as f64 * 100.0;
    if pct < 10.0 {
        check.warnings.push(format!("Low overlap: only {pct:.1}% of keys match"));
    }
    check
}

/// Packages present in both releases, and the ones in only one.
#[derive(Debug)]
pub struct MergedReleases {
    pub matched: StringTable,
    pub unmatched: StringTable,
}

/// Row numbers in each table that share one `source` key.
#[derive(Default)]
struct KeyRows {
    left: Vec<usize>,
    right: Vec<usize>,
}

/// Outer-join two releases on `source`, the way pandas' `merge(how="outer")` does: keys in
/// sorted order, every pairing for a repeated key, the second release's copies of shared
/// columns dropped, and `homepage` renamed to `upstream_repo_url`.
pub fn merge_release_packages(tables: &[StringTable]) -> Result<MergedReleases> {
    let [a, b] = tables else {
        bail!("Expected exactly 2 DataFrames, got {}", tables.len());
    };
    let shared: Vec<String> = a.names().iter().filter(|n| *n != "source" && b.has_column(n)).cloned().collect();
    let mut b = b.clone();
    b.drop_columns(&shared);
    info!("Dropped redundant columns from second DataFrame before merge: {shared:?}");

    let check = validate_merge_safety(a, &b, "source");
    if !check.is_safe {
        bail!("Cannot safely merge DataFrames:\n  - {}", check.warnings.join("\n  - "));
    }
    for w in &check.warnings {
        warn!("Merge validation: {w}");
    }

    let a_names = a.names().to_vec();
    let b_extra: Vec<String> = b.names().iter().filter(|n| *n != "source").cloned().collect();
    let names: Vec<String> = a_names.iter().chain(&b_extra).cloned().collect();
    let a_source = a_names.iter().position(|n| n == "source").expect("validated");
    let b_extra_idx: Vec<usize> =
        b_extra.iter().map(|n| b.names().iter().position(|m| m == n).expect("own column")).collect();

    // Rows per key; a missing key sorts last, as pandas puts NaN last.
    let mut keys: BTreeMap<(bool, Option<String>), KeyRows> = BTreeMap::new();
    let sort_key = |k: &Option<String>| (k.is_none(), k.clone());
    for (i, k) in a.column("source").unwrap_or_default().iter().enumerate() {
        keys.entry(sort_key(k)).or_default().left.push(i);
    }
    for (i, k) in b.column("source").unwrap_or_default().iter().enumerate() {
        keys.entry(sort_key(k)).or_default().right.push(i);
    }

    let mut matched = StringTable::with_columns(&names);
    let mut unmatched = StringTable::with_columns(&names);
    for ((_, key), KeyRows { left: rows_a, right: rows_b }) in &keys {
        let b_part = |j: usize| -> Vec<Option<String>> {
            let row = b.row(j);
            b_extra_idx.iter().map(|&c| row[c].clone()).collect()
        };
        match (rows_a.is_empty(), rows_b.is_empty()) {
            (false, false) => {
                for &i in rows_a {
                    for &j in rows_b {
                        let mut row = a.row(i);
                        row.extend(b_part(j));
                        matched.push_row(row);
                    }
                }
            }
            (false, true) => {
                for &i in rows_a {
                    let mut row = a.row(i);
                    row.extend(std::iter::repeat_n(None, b_extra.len()));
                    unmatched.push_row(row);
                }
            }
            (true, false) => {
                for &j in rows_b {
                    let mut row = vec![None; a_names.len()];
                    row[a_source] = key.clone();
                    row.extend(b_part(j));
                    unmatched.push_row(row);
                }
            }
            (true, true) => unreachable!("every key has a row"),
        }
    }
    matched.rename("homepage", "upstream_repo_url");
    unmatched.rename("homepage", "upstream_repo_url");
    Ok(MergedReleases { matched, unmatched })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> Option<String> {
        Some(v.to_string())
    }

    #[test]
    fn deb822_fields() {
        let f = parse_deb822(
            "Package: foo\nBinary: a,\n b\nVersion:  1.0-1  \nhomepage: https://x\nDepends:\n  libc6,\n .\n# c\nnot a field\n",
        );
        assert_eq!(f["package"], "foo");
        assert_eq!(f["binary"], "a,\n b");
        assert_eq!(f["version"], "1.0-1");
        assert_eq!(f["homepage"], "https://x");
        assert_eq!(f["depends"], "\n  libc6,\n .");
        assert!(!f.contains_key("maintainer"));
        assert_eq!(parse_deb822("Package: a\nPackage: b\n")["package"], "b");
    }

    #[test]
    fn sources_file() {
        let text = b"Package: one\nVersion: 1.0-1\nHomepage: https://github.com/a/one\n\n  \t\nSource: two\nVersion: 2:3.4-1\n\n\n";
        let t = parse_sources(&text[..], "trixie").unwrap();
        assert_eq!(t.names(), ["source", "trixie_version", "homepage", "depends", "maintainer"]);
        assert_eq!(t.column("source").unwrap(), [s("one"), s("two")]);
        assert_eq!(t.column("homepage").unwrap(), [s("https://github.com/a/one"), None]);
    }

    #[test]
    fn invalid_utf8_is_dropped() {
        assert_eq!(decode_utf8_ignore(b"a\xffb"), "ab");
        assert_eq!(python_splitlines("a\r\nb\rc\x0cd\n"), ["a", "b", "c", "d"]);
    }

    #[test]
    fn github_filter() {
        let mut t = StringTable::with_columns(&["source", "homepage"]);
        for (src, home) in [
            ("a", s("https://github.com/o/a.git")),
            ("b", s("https://gitlab.com/o/b")),
            ("c", None),
            ("d", s("http://github.com/o/a")),
            ("e", s("https://githubXcom/o/e")),
            ("f", s("git@github.com:o/f.git")),
        ] {
            t.push_row(vec![s(src), home]);
        }
        let f = filter_github_repos(&t).unwrap();
        assert_eq!(f.column("source").unwrap(), [s("a")]);
        assert_eq!(f.column("homepage").unwrap(), [s("https://github.com/o/a")]);
    }

    fn release(rel: &str, rows: &[(&str, &str)]) -> StringTable {
        let mut t = StringTable::with_columns(&["source", &format!("{rel}_version"), "homepage"]);
        for (src, v) in rows {
            t.push_row(vec![s(src), s(v), s(&format!("https://github.com/o/{src}"))]);
        }
        t
    }

    #[test]
    fn merge_matches_pandas_outer() {
        let a = release("bookworm", &[("zed", "1"), ("pkg", "1"), ("only_a", "1")]);
        let b = release("trixie", &[("pkg", "2"), ("only_b", "2"), ("zed", "2")]);
        let m = merge_release_packages(&[a, b]).unwrap();
        assert_eq!(m.matched.names(), ["source", "bookworm_version", "upstream_repo_url", "trixie_version"]);
        assert_eq!(m.matched.column("source").unwrap(), [s("pkg"), s("zed")]);
        assert_eq!(m.matched.column("trixie_version").unwrap(), [s("2"), s("2")]);
        assert_eq!(m.unmatched.column("source").unwrap(), [s("only_a"), s("only_b")]);
        assert_eq!(m.unmatched.column("bookworm_version").unwrap(), [s("1"), None]);
        assert_eq!(m.unmatched.column("upstream_repo_url").unwrap(), [s("https://github.com/o/only_a"), None]);
        assert_eq!(m.unmatched.column("trixie_version").unwrap(), [None, s("2")]);
    }

    #[test]
    fn merge_needs_two_tables() {
        assert!(merge_release_packages(&[release("a", &[])]).is_err());
    }

    #[test]
    fn upstream_column() {
        let mut t = release("trixie", &[("a", "2:1.2.3-4"), ("b", "")]);
        add_upstream_version_column(&mut t, "trixie_version", "trixie_upstream_version").unwrap();
        assert_eq!(t.column("trixie_upstream_version").unwrap(), [s("1.2.3"), None]);
        assert!(add_upstream_version_column(&mut t, "nope", "x").is_err());
    }
}
