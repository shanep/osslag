//! Cloning upstream repos and reading their commit history with libgit2.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Once};

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::TimestampMicrosecondArray;
use arrow::array::{ArrayRef, Int64Array, LargeStringArray, ListBuilder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use git2::{Cred, DiffOptions, FetchOptions, ObjectType, RemoteCallbacks, Repository, Sort};
use tracing::warn;

use crate::github::GitHubApi;
use crate::vcs::url::{extract_owner_name_repo, normalize_https_repo_url};

const SERVER_CONNECT_TIMEOUT_MS: i32 = 10_000;
const SERVER_TIMEOUT_MS: i32 = 300_000;

/// One row of `{distro}_all_upstream_commits.parquet`, before `repo_url` and `source` are added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitRow {
    pub hash: String,
    pub author: Option<String>,
    pub email: Option<String>,
    pub message: Option<String>,
    /// Committer time in Unix seconds.
    pub timestamp: i64,
    pub files: Vec<String>,
    pub tags: Vec<String>,
}

/// libgit2's timeouts are process-wide, so set them once.
fn set_timeouts() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: called once, before any clone starts on another thread.
        unsafe {
            if let Err(e) = git2::opts::set_server_connect_timeout_in_milliseconds(SERVER_CONNECT_TIMEOUT_MS) {
                warn!("could not set the git connect timeout: {e}");
            }
            if let Err(e) = git2::opts::set_server_timeout_in_milliseconds(SERVER_TIMEOUT_MS) {
                warn!("could not set the git server timeout: {e}");
            }
        }
    });
}

/// Fetch options that log in with `GITHUB_USERNAME` and `GITHUB_TOKEN` when both are set.
fn fetch_options<'a>() -> FetchOptions<'a> {
    let mut callbacks = RemoteCallbacks::new();
    if let (Ok(user), Ok(token)) = (std::env::var("GITHUB_USERNAME"), std::env::var("GITHUB_TOKEN"))
        && !user.is_empty()
        && !token.is_empty()
    {
        callbacks.credentials(move |_, _, _| Cred::userpass_plaintext(&user, &token));
    }
    let mut options = FetchOptions::new();
    options.remote_callbacks(callbacks);
    options
}

/// Write `{dest}.failed` next to the clone target, so later runs skip it.
fn mark_failed(dest: &Path, reason: &str) {
    let Some(name) = dest.file_name() else { return };
    let marker = dest.with_file_name(format!("{}.failed", name.to_string_lossy()));
    if let Err(e) = std::fs::write(&marker, reason) {
        warn!("could not write {}: {e}", marker.display());
    }
}

/// Clone a GitHub repo into `dest` with its tags. An existing checkout counts as success.
/// A failure removes any partial clone and leaves a `{dest}.failed` marker saying which stage failed.
pub fn clone_repo(api: &dyn GitHubApi, repo_url: &str, dest: &Path, branch: Option<&str>) -> Result<(), String> {
    let normalized = normalize_https_repo_url(Some(repo_url));
    let Some(url) = normalized.url else {
        let error = normalized.error.unwrap_or_else(|| "Failed to normalize repository URL".into());
        warn!("Failed to normalize URL {repo_url}: {error}");
        mark_failed(dest, &format!("normalize: {error}"));
        return Err(error);
    };
    let Some(repo) = extract_owner_name_repo(&url) else {
        let error = "Failed to extract owner/repo from URL".to_string();
        warn!("Failed to extract owner/repo from {url}");
        mark_failed(dest, &format!("extract: {error}"));
        return Err(error);
    };
    if dest.join(".git").exists() {
        return Ok(());
    }
    if let Err(error) = api.check_repo_exists(&repo.owner, &repo.name) {
        mark_failed(dest, &format!("gh_check: {error} url: {url}"));
        return Err(error);
    }
    set_timeouts();
    let cloned = (|| -> Result<(), git2::Error> {
        let repo = git2::build::RepoBuilder::new().fetch_options(fetch_options()).clone(&url, dest)?;
        if let Some(branch) = branch {
            let (object, reference) = repo.revparse_ext(branch)?;
            repo.checkout_tree(&object, None)?;
            match reference.and_then(|r| r.name().ok().map(str::to_string)) {
                Some(name) => repo.set_head(&name)?,
                None => repo.set_head_detached(object.id())?,
            }
        }
        repo.find_remote("origin")?.fetch(&[] as &[&str], Some(&mut fetch_options()), None)
    })();
    cloned.map_err(|e| {
        if dest.exists()
            && let Err(rm) = std::fs::remove_dir_all(dest)
        {
            warn!("could not remove partial clone {}: {rm}", dest.display());
        }
        let error = format!("git error: {}", e.message());
        mark_failed(dest, &format!("clone: {error}"));
        error
    })
}

/// Python's `str.strip()`: Unicode whitespace plus the ASCII separators \x1c to \x1f.
fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

/// Decode like pygit2 (invalid UTF-8 becomes U+FFFD), then strip. Empty stays empty.
fn text(bytes: &[u8]) -> String {
    py_strip(&String::from_utf8_lossy(bytes)).to_string()
}

/// Paths a commit touched, compared with its first parent or, for the root commit, an empty tree.
/// Merge commits get no files.
fn changed_paths(repo: &Repository, commit: &git2::Commit<'_>) -> Vec<String> {
    if commit.parent_count() > 1 {
        return Vec::new();
    }
    let diff = (|| -> Result<git2::Diff<'_>, git2::Error> {
        let tree = commit.tree()?;
        let parent_tree = match commit.parent_count() {
            0 => None,
            _ => Some(commit.parent(0)?.tree()?),
        };
        let mut options = DiffOptions::new();
        options.skip_binary_check(true).ignore_submodules(true).context_lines(0);
        repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), Some(&mut options))
    })();
    let Ok(diff) = diff else { return Vec::new() };
    diff.deltas()
        .map(|d| {
            let new = d.new_file().path_bytes().filter(|p| !p.is_empty());
            String::from_utf8_lossy(new.or(d.old_file().path_bytes()).unwrap_or_default()).into_owned()
        })
        .collect()
}

/// Tag names for each commit id, with annotated tags peeled to their commit.
fn tags_by_commit(repo: &Repository) -> Result<HashMap<git2::Oid, Vec<String>>> {
    let mut tags: HashMap<git2::Oid, Vec<String>> = HashMap::new();
    for reference in repo.references_glob("refs/tags/*")? {
        let Ok(reference) = reference else { continue };
        let Some(name) = reference.name().ok().and_then(|n| n.strip_prefix("refs/tags/")) else { continue };
        if let Ok(object) = reference.peel(ObjectType::Any)
            && object.kind() == Some(ObjectType::Commit)
        {
            tags.entry(object.id()).or_default().push(name.to_string());
        }
    }
    Ok(tags)
}

/// Every commit reachable from `branch` (or HEAD), oldest first. With `since` (Unix seconds),
/// commits with an earlier committer time are left out.
pub fn load_commits(repo_path: &Path, branch: Option<&str>, since: Option<i64>) -> Result<Vec<CommitRow>> {
    if !repo_path.exists() {
        bail!("Repository path does not exist: {}", repo_path.display());
    }
    if !repo_path.join(".git").exists() {
        bail!("Not a Git repository (missing .git directory): {}", repo_path.display());
    }
    let repo = Repository::open(repo_path).with_context(|| format!("opening {}", repo_path.display()))?;
    let start = match branch {
        Some(b) => repo
            .find_reference(&format!("refs/heads/{b}"))
            .or_else(|_| repo.find_reference(&format!("refs/remotes/origin/{b}")))
            .map_err(|_| anyhow!("Branch '{b}' not found in {}", repo_path.display()))?
            .peel_to_commit()?
            .id(),
        None => repo
            .head()
            .and_then(|h| h.peel_to_commit())
            .map_err(|_| anyhow!("Repository has no HEAD (empty repository?): {}", repo_path.display()))?
            .id(),
    };
    let tags = tags_by_commit(&repo)?;
    let mut walk = repo.revwalk()?;
    walk.set_sorting(Sort::TOPOLOGICAL | Sort::TIME)?;
    walk.push(start)?;
    let mut rows = Vec::new();
    for oid in walk {
        let commit = repo.find_commit(oid?)?;
        let timestamp = commit.time().seconds();
        if since.is_some_and(|s| timestamp < s) {
            continue;
        }
        let author = commit.author();
        rows.push(CommitRow {
            hash: oid_hex(commit.id()),
            author: Some(text(author.name_bytes())),
            email: Some(text(author.email_bytes())),
            message: Some(text(commit.message_bytes())),
            timestamp,
            files: changed_paths(&repo, &commit),
            tags: tags.get(&commit.id()).cloned().unwrap_or_default(),
        });
    }
    rows.sort_by_key(|r| r.timestamp);
    Ok(rows)
}

fn oid_hex(oid: git2::Oid) -> String {
    oid.to_string()
}

fn string_list(rows: &[CommitRow], f: impl Fn(&CommitRow) -> &[String]) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for row in rows {
        for v in f(row) {
            builder.values().append_value(v);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

/// Commit rows as a record batch with the columns Python wrote, plus `repo_url` and `source`.
pub fn commits_to_batch(rows: &[CommitRow], repo_url: &str, source: &str) -> Result<RecordBatch> {
    let s = |f: &dyn Fn(&CommitRow) -> Option<&str>| -> ArrayRef {
        Arc::new(rows.iter().map(f).collect::<LargeStringArray>())
    };
    let columns: Vec<(&str, ArrayRef)> = vec![
        ("hash", s(&|r| Some(&r.hash))),
        ("author", s(&|r| r.author.as_deref())),
        ("email", s(&|r| r.email.as_deref())),
        ("message", s(&|r| r.message.as_deref())),
        ("timestamp", Arc::new(rows.iter().map(|r| Some(r.timestamp)).collect::<Int64Array>())),
        (
            "date",
            Arc::new(
                rows.iter()
                    .map(|r| Some(r.timestamp * 1_000_000))
                    .collect::<TimestampMicrosecondArray>()
                    .with_timezone("UTC"),
            ),
        ),
        ("files", string_list(rows, |r| &r.files)),
        ("tags", string_list(rows, |r| &r.tags)),
        ("repo_url", Arc::new(LargeStringArray::from(vec![repo_url; rows.len()]))),
        ("source", Arc::new(LargeStringArray::from(vec![source; rows.len()]))),
    ];
    let fields: Vec<Field> = columns.iter().map(|(n, c)| Field::new(*n, c.data_type().clone(), true)).collect();
    debug_assert!(matches!(fields[6].data_type(), DataType::List(_)));
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns.into_iter().map(|(_, c)| c).collect())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use git2::{Signature, Time};

    struct FakeGitHub(Result<(), String>);

    impl GitHubApi for FakeGitHub {
        fn check_repo_exists(&self, _: &str, _: &str) -> Result<(), String> {
            self.0.clone()
        }
    }

    /// Commit `files` (path, contents) on top of HEAD at time `t`.
    fn commit(repo: &Repository, t: i64, msg: &str, files: &[(&str, &str)]) -> git2::Oid {
        let workdir = repo.workdir().unwrap();
        let mut index = repo.index().unwrap();
        for (path, body) in files {
            let full = workdir.join(path);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(full, body).unwrap();
            index.add_path(Path::new(path)).unwrap();
        }
        index.write().unwrap();
        let tree = repo.find_tree(index.write_tree().unwrap()).unwrap();
        let sig = Signature::new("Ann  ", "ann@x", &Time::new(t, 0)).unwrap();
        let parents: Vec<git2::Commit<'_>> =
            repo.head().ok().map(|h| h.peel_to_commit().unwrap()).into_iter().collect();
        let parent_refs: Vec<&git2::Commit<'_>> = parents.iter().collect();
        repo.commit(Some("HEAD"), &sig, &sig, msg, &tree, &parent_refs).unwrap()
    }

    #[test]
    fn loads_history_files_and_tags() {
        let dir = tempfile::tempdir().unwrap();
        let repo = Repository::init(dir.path()).unwrap();
        let first = commit(&repo, 1_000, "  first\n", &[("README.md", "a"), ("src/a.c", "x")]);
        let second = commit(&repo, 2_000, "second", &[("src/a.c", "y")]);
        repo.tag_lightweight("v1", &repo.find_object(first, None).unwrap(), false).unwrap();
        let sig = Signature::new("T", "t@x", &Time::new(3_000, 0)).unwrap();
        repo.tag("v2", &repo.find_object(second, None).unwrap(), &sig, "annotated", false).unwrap();

        let rows = load_commits(dir.path(), None, None).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].message.as_deref(), Some("first"));
        assert_eq!(rows[0].author.as_deref(), Some("Ann"));
        assert_eq!(rows[0].files, vec!["README.md", "src/a.c"]);
        assert_eq!(rows[0].tags, vec!["v1"]);
        assert_eq!(rows[1].files, vec!["src/a.c"]);
        assert_eq!(rows[1].tags, vec!["v2"]);
        assert_eq!(rows[1].hash, second.to_string());

        let recent = load_commits(dir.path(), None, Some(1_500)).unwrap();
        assert_eq!(recent.len(), 1);

        let batch = commits_to_batch(&rows, "https://github.com/o/r", "pkg").unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 10);
    }

    #[test]
    fn missing_repo_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_commits(&dir.path().join("nope"), None, None).is_err());
        assert!(load_commits(dir.path(), None, None).unwrap_err().to_string().contains("missing .git"));
    }

    #[test]
    fn clone_failures_leave_markers() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("o--r");
        let marker = dir.path().join("o--r.failed");

        let err = clone_repo(&FakeGitHub(Err("Not Found".into())), "https://github.com/o/r", &dest, None);
        assert_eq!(err, Err("Not Found".to_string()));
        assert_eq!(std::fs::read_to_string(&marker).unwrap(), "gh_check: Not Found url: https://github.com/o/r");

        let err = clone_repo(&FakeGitHub(Ok(())), "https://gitlab.com/o/r", &dest, None);
        assert!(err.is_err());
        assert!(std::fs::read_to_string(&marker).unwrap().starts_with("normalize: "));
    }

    #[test]
    fn existing_checkout_is_success() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("o--r");
        Repository::init(&dest).unwrap();
        assert_eq!(clone_repo(&FakeGitHub(Err("unused".into())), "https://github.com/o/r", &dest, None), Ok(()));
    }

    #[test]
    fn python_strip() {
        assert_eq!(py_strip("\x1c a \u{2028}"), "a");
    }
}
