//! Linux distribution package data, behind a handler per distribution.

pub mod debian;
pub mod version;

use std::path::Path;

use anyhow::{Result, bail};

use crate::io::table::StringTable;

/// What every distribution module provides to the pipeline.
pub trait DistroHandler: Sync {
    fn fetch_packages(&self, release: &str) -> Result<StringTable>;
    fn filter_github_repos(&self, packages: &StringTable) -> Result<StringTable>;
    fn add_local_repo_cache_path_column(&self, packages: &mut StringTable, cache_dir: &Path) -> Result<()>;
    fn add_upstream_version_column(
        &self,
        packages: &mut StringTable,
        version_column: &str,
        new_column: &str,
    ) -> Result<()>;
    fn merge_release_packages(&self, tables: &[StringTable]) -> Result<debian::MergedReleases>;
}

pub struct Debian;

impl DistroHandler for Debian {
    fn fetch_packages(&self, release: &str) -> Result<StringTable> {
        debian::fetch_packages(release)
    }
    fn filter_github_repos(&self, packages: &StringTable) -> Result<StringTable> {
        debian::filter_github_repos(packages)
    }
    fn add_local_repo_cache_path_column(&self, packages: &mut StringTable, cache_dir: &Path) -> Result<()> {
        debian::add_local_repo_cache_path_column(packages, cache_dir)
    }
    fn add_upstream_version_column(
        &self,
        packages: &mut StringTable,
        version_column: &str,
        new_column: &str,
    ) -> Result<()> {
        debian::add_upstream_version_column(packages, version_column, new_column)
    }
    fn merge_release_packages(&self, tables: &[StringTable]) -> Result<debian::MergedReleases> {
        debian::merge_release_packages(tables)
    }
}

/// Distributions with a handler, sorted.
pub fn supported_distros() -> Vec<&'static str> {
    vec!["debian"]
}

/// The handler for a distribution name, ignoring case.
pub fn get_handler(distro: &str) -> Result<&'static dyn DistroHandler> {
    match distro.to_lowercase().as_str() {
        "debian" => Ok(&Debian),
        _ => bail!("Unsupported distro '{distro}'. Supported: {}", supported_distros().join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handlers() {
        assert!(get_handler("Debian").is_ok());
        assert!(get_handler("fedora").err().unwrap().to_string().contains("Supported: debian"));
    }
}
