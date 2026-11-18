//! Trivial (documentation-only) commits, which MALTA leaves out of development activity.

/// The suffix of the last path component, like Python's `PurePosixPath(path).suffix`.
fn posix_suffix(path: &str) -> &str {
    let name = path.trim_end_matches('/').rsplit('/').next().unwrap_or("");
    match name.rfind('.') {
        Some(i) if i > 0 && i < name.len() - 1 => &name[i..],
        _ => "",
    }
}

/// A commit is trivial when it touched no files (merge commits included) or only Markdown files.
pub fn is_trivial_commit<S: AsRef<str>>(files: &[S]) -> bool {
    files.iter().all(|f| posix_suffix(f.as_ref()).to_lowercase() == ".md")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_only_is_trivial() {
        assert!(is_trivial_commit::<&str>(&[]));
        assert!(is_trivial_commit(&["README.md"]));
        assert!(is_trivial_commit(&["docs/a.MD", "b.md"]));
        assert!(!is_trivial_commit(&["a.md", "src/x.rs"]));
        assert!(!is_trivial_commit(&["notes.markdown"]));
        assert!(!is_trivial_commit(&[".md"]));
        assert!(!is_trivial_commit(&["Makefile"]));
    }
}
