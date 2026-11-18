//! GitHub repository URLs.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;

/// Result of normalizing a repository URL. SSH URLs give neither a URL nor an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedUrl {
    pub url: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoOwnerName {
    pub owner: String,
    pub name: String,
}

static SSH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^git@github\.com:([\w.-]+)/([\w.-]+)$").expect("ssh regex"));
static HTTPS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^https://github\.com/([\w.-]+)/([\w.-]+)").expect("https regex"));

/// Normalize a GitHub URL to `https://github.com/{owner}/{repo}`.
///
/// Strips whitespace, any `#fragment`, a `.git` suffix and one trailing slash, upgrades `http`
/// to `https`, and drops anything after the repo name (`/tree/main`).
pub fn normalize_https_repo_url(url: Option<&str>) -> NormalizedUrl {
    let err = |e: &str| NormalizedUrl { url: None, error: Some(e.to_string()) };
    let Some(url) = url.filter(|u| !u.is_empty()) else {
        return err("Invalid URL: None or not a string");
    };
    let mut url = url.trim();
    if let Some((before, _)) = url.split_once('#') {
        url = before;
    }
    let url = match url.strip_prefix("http://") {
        Some(rest) => format!("https://{rest}"),
        None => url.to_string(),
    };
    let mut matching = url.strip_suffix(".git").unwrap_or(&url);
    matching = matching.strip_suffix('/').unwrap_or(matching);

    if SSH.is_match(matching) {
        return NormalizedUrl { url: None, error: None };
    }
    if let Some(caps) = HTTPS.captures(matching) {
        return NormalizedUrl { url: Some(format!("https://github.com/{}/{}", &caps[1], &caps[2])), error: None };
    }
    err("URL does not match expected git repository patterns")
}

/// Owner and repo name of a GitHub URL, or `None` if it does not normalize.
pub fn extract_owner_name_repo(url: &str) -> Option<RepoOwnerName> {
    let normalized = normalize_https_repo_url(Some(url)).url?;
    let mut parts = normalized.rsplit('/');
    let name = parts.next()?.to_string();
    let owner = parts.next()?.to_string();
    Some(RepoOwnerName { owner, name })
}

/// Local clone directory for a repo: `{REPOS_CACHE_DIR or repos_dir}/{owner}--{repo}`.
/// With `must_exist`, returns `None` unless the directory is already there.
pub fn construct_repo_local_path(url: &str, repos_dir: &Path, must_exist: bool) -> Option<PathBuf> {
    let repo = extract_owner_name_repo(url)?;
    let base = std::env::var_os("REPOS_CACHE_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| repos_dir.to_path_buf());
    let path = base.join(format!("{}--{}", repo.owner, repo.name));
    if path.exists() {
        return Some(std::fs::canonicalize(&path).unwrap_or(path));
    }
    (!must_exist).then_some(path)
}

/// Hash of the first commit tagged with `version` as `v1.2.3`, `1.2.3`, `release-1.2.3` or
/// `version_1.2.3` (the separator is optional).
pub fn find_upstream_version_tag_commit<'a>(
    commits: impl IntoIterator<Item = (&'a str, &'a [String])>,
    version: &str,
) -> Option<String> {
    let v = regex::escape(version);
    let patterns: Vec<Regex> =
        [format!("^v{v}$"), format!("^{v}$"), format!("^release[-_]?{v}$"), format!("^version[-_]?{v}$")]
            .iter()
            .map(|p| Regex::new(p).expect("escaped version regex"))
            .collect();
    commits
        .into_iter()
        .find(|(_, tags)| tags.iter().any(|t| patterns.iter().any(|p| p.is_match(t))))
        .map(|(hash, _)| hash.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(url: &str) -> NormalizedUrl {
        NormalizedUrl { url: Some(url.to_string()), error: None }
    }

    #[test]
    fn normalizes() {
        assert_eq!(normalize_https_repo_url(Some("http://github.com/foo/bar.git")), ok("https://github.com/foo/bar"));
        assert_eq!(
            normalize_https_repo_url(Some("https://github.com/foo/bar/tree/x")),
            ok("https://github.com/foo/bar")
        );
        assert_eq!(
            normalize_https_repo_url(Some(" https://github.com/foo/bar#readme ")),
            ok("https://github.com/foo/bar")
        );
        assert_eq!(
            normalize_https_repo_url(Some("git@github.com:foo/bar.git")),
            NormalizedUrl { url: None, error: None }
        );
        assert!(normalize_https_repo_url(Some("https://gitlab.com/foo/bar")).error.is_some());
        assert!(normalize_https_repo_url(None).error.is_some());
    }

    #[test]
    fn owner_and_name() {
        let r = extract_owner_name_repo("https://github.com/sboudrias/Inquirer.js").unwrap();
        assert_eq!((r.owner.as_str(), r.name.as_str()), ("sboudrias", "Inquirer.js"));
    }

    #[test]
    fn version_tags() {
        let tags_a = vec!["foo".to_string()];
        let tags_b = vec!["release_1.2.3".to_string()];
        let commits = [("a", tags_a.as_slice()), ("b", tags_b.as_slice())];
        assert_eq!(find_upstream_version_tag_commit(commits, "1.2.3"), Some("b".into()));
        assert_eq!(find_upstream_version_tag_commit(commits, "1.2"), None);
        assert_eq!(find_upstream_version_tag_commit(commits, "1x2.3"), None);
    }
}
