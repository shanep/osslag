//! Parity with the Python implementation, using the golden fixtures in tests/fixtures/parity/
//! (written by scripts/gen_parity_fixtures.py before the Python code was retired).

use std::path::PathBuf;

use chrono::DateTime;
use osslag::distro::version::extract_upstream_version;
use osslag::io::columns::{bools, floats, ints, strings};
use osslag::io::read_parquet;
use osslag::metrics::frames::{RepoTables, results_to_batch};
use osslag::metrics::malta::{Constants, Package, score_repos};
use osslag::metrics::pvac::{categorize_development_activity, lookup_category};
use osslag::vcs::trivial::is_trivial_commit;
use osslag::vcs::url::normalize_https_repo_url;
use serde::Deserialize;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity").join(name)
}

#[derive(Deserialize)]
struct Pvac {
    category: Option<String>,
    epoch: Option<u64>,
    major: Option<u64>,
    minor: Option<u64>,
    patch: Option<u64>,
}

#[derive(Deserialize)]
struct StringCase {
    input: String,
    upstream_version: Option<String>,
    pvac: Pvac,
    normalized_url: Option<String>,
    normalize_error: Option<String>,
}

#[derive(Deserialize)]
struct ActivityCase {
    a: String,
    b: String,
    activity: String,
}

#[derive(Deserialize)]
struct Strings {
    cases: Vec<StringCase>,
    activity: Vec<ActivityCase>,
}

#[test]
fn string_functions_match_python() {
    let data: Strings = serde_json::from_str(&std::fs::read_to_string(fixture("strings.json")).unwrap()).unwrap();
    let mut failures = Vec::new();
    for c in &data.cases {
        let input = c.input.as_str();
        if extract_upstream_version(input) != c.upstream_version {
            failures.push(format!(
                "upstream {input:?}: {:?} != {:?}",
                extract_upstream_version(input),
                c.upstream_version
            ));
        }
        let got = lookup_category(input);
        let got_tuple = got.map(|v| (v.category.as_str().to_string(), v.epoch, v.major, v.minor, v.patch));
        let want_tuple = c.pvac.category.clone().map(|cat| {
            (cat, c.pvac.epoch.unwrap(), c.pvac.major.unwrap(), c.pvac.minor.unwrap(), c.pvac.patch.unwrap())
        });
        if got_tuple != want_tuple {
            failures.push(format!("pvac {input:?}: {got_tuple:?} != {want_tuple:?}"));
        }
        let norm = normalize_https_repo_url(Some(input));
        if norm.url != c.normalized_url || norm.error != c.normalize_error {
            failures.push(format!("url {input:?}: {norm:?} != {:?} {:?}", c.normalized_url, c.normalize_error));
        }
    }
    for c in &data.activity {
        let got = categorize_development_activity(&c.a, &c.b);
        if got != c.activity {
            failures.push(format!("activity {:?} {:?}: {got} != {}", c.a, c.b, c.activity));
        }
    }
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[derive(Deserialize)]
struct TrivialCase {
    files: Vec<String>,
    is_trivial: bool,
}

#[test]
fn trivial_labels_match_python() {
    let cases: Vec<TrivialCase> =
        serde_json::from_str(&std::fs::read_to_string(fixture("trivial.json")).unwrap()).unwrap();
    for c in cases {
        assert_eq!(is_trivial_commit(&c.files), c.is_trivial, "{:?}", c.files);
    }
}

#[test]
fn malta_scores_match_python_exactly() {
    let commits = read_parquet(&fixture("malta_commits.parquet")).unwrap();
    let prs = read_parquet(&fixture("malta_prs.parquet")).unwrap();
    let meta = read_parquet(&fixture("malta_meta.parquet")).unwrap();
    let packages_batch = read_parquet(&fixture("malta_packages.parquet")).unwrap();
    let expected = read_parquet(&fixture("malta_expected.parquet")).unwrap();
    let eval_end =
        DateTime::parse_from_rfc3339(std::fs::read_to_string(fixture("malta_eval_end.txt")).unwrap().trim()).unwrap();

    let tables = RepoTables::new(&commits, Some(&prs), Some(&meta)).unwrap();
    let packages: Vec<Package> = strings(&packages_batch, "source")
        .unwrap()
        .into_iter()
        .zip(strings(&packages_batch, "repo_url").unwrap())
        .map(|(s, u)| Package { source: s.unwrap(), repo_url: u.unwrap() })
        .collect();
    let results = score_repos(&packages, |url| tables.repo_data(url), eval_end, Constants::default()).unwrap();
    let got = results_to_batch(&results).unwrap();

    assert_eq!(got.num_rows(), expected.num_rows());
    let names: Vec<String> = expected.schema().fields().iter().map(|f| f.name().clone()).collect();
    let got_names: Vec<String> = got.schema().fields().iter().map(|f| f.name().clone()).collect();
    assert_eq!(got_names, names, "column order");

    let mut failures = Vec::new();
    for name in &names {
        let field = expected.schema().field_with_name(name).unwrap().clone();
        match field.data_type() {
            arrow::datatypes::DataType::Float64 => {
                let (g, e) = (floats(&got, name).unwrap(), floats(&expected, name).unwrap());
                for (row, (g, e)) in g.iter().zip(&e).enumerate() {
                    let same = match (g, e) {
                        (Some(g), Some(e)) => g.to_bits() == e.to_bits() || (g.is_nan() && e.is_nan()),
                        (None, Some(e)) => e.is_nan(),
                        (g, e) => g == e,
                    };
                    if !same {
                        failures.push(format!("{name} row {row}: {g:?} != {e:?}"));
                    }
                }
            }
            arrow::datatypes::DataType::Int64 => {
                if ints(&got, name).unwrap() != ints(&expected, name).unwrap() {
                    failures.push(format!("{name} differs"));
                }
            }
            arrow::datatypes::DataType::Boolean => {
                if bools(&got, name).unwrap() != bools(&expected, name).unwrap() {
                    failures.push(format!("{name} differs"));
                }
            }
            _ => {
                if strings(&got, name).unwrap() != strings(&expected, name).unwrap() {
                    failures.push(format!("{name} differs"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.iter().take(40).cloned().collect::<Vec<_>>().join("\n")
    );
}
