//! Compare two parquet files: schema, row count, then every value after sorting rows by key.
//!
//! cargo run --release --example parquet_diff -- A.parquet B.parquet [--key a,b] [--ignore c,d]
//!
//! Exits 0 when the files match, 1 when they differ and 2 on bad arguments.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use arrow::array::RecordBatch;
use arrow::datatypes::DataType;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use osslag::io::read_parquet;

const SHOW: usize = 10;

struct Args {
    left: String,
    right: String,
    keys: Vec<String>,
    ignore: HashSet<String>,
}

fn list(value: Option<String>) -> Vec<String> {
    value.map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_default()
}

fn parse_args() -> Result<Args> {
    let mut files = Vec::new();
    let (mut keys, mut ignore) = (None, None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--key" => keys = args.next(),
            "--ignore" => ignore = args.next(),
            _ => files.push(arg),
        }
    }
    let [left, right] = <[String; 2]>::try_from(files).ok().context("need two parquet files")?;
    Ok(Args { left, right, keys: list(keys), ignore: list(ignore).into_iter().collect() })
}

/// Every row as strings, in the given column order.
fn rows(batch: &RecordBatch, columns: &[String]) -> Result<Vec<Vec<String>>> {
    let options = FormatOptions::default().with_null("<null>");
    let formatters = columns
        .iter()
        .map(|c| ArrayFormatter::try_new(batch.column_by_name(c).expect("shared column").as_ref(), &options))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((0..batch.num_rows()).map(|r| formatters.iter().map(|f| f.value(r).to_string()).collect()).collect())
}

/// A type name without list item names, which differ between writers (`item`, `element`)
/// and mean nothing to readers.
fn type_name(t: &DataType) -> String {
    match t {
        DataType::List(f) | DataType::LargeList(f) => format!("List({})", type_name(f.data_type())),
        other => other.to_string(),
    }
}

fn compare(args: &Args) -> Result<usize> {
    let left = read_parquet(Path::new(&args.left))?;
    let right = read_parquet(Path::new(&args.right))?;
    let mut problems = 0;

    let types = |b: &RecordBatch| -> BTreeMap<String, String> {
        b.schema().fields().iter().map(|f| (f.name().clone(), type_name(f.data_type()))).collect()
    };
    let (lt, rt) = (types(&left), types(&right));
    for (name, ty) in &lt {
        if args.ignore.contains(name) {
            continue;
        }
        match rt.get(name) {
            None => {
                println!("column {name} only in left ({ty})");
                problems += 1;
            }
            // pandas writes a column of nothing but nulls as type null. The values still get compared.
            Some(other) if ty == "Null" || other == "Null" => {
                println!("note: column {name} is all null on one side ({ty} vs {other})");
            }
            Some(other) if other != ty => {
                println!("column {name}: left {ty}, right {other}");
                problems += 1;
            }
            Some(_) => {}
        }
    }
    for (name, ty) in rt.iter().filter(|(n, _)| !lt.contains_key(*n) && !args.ignore.contains(*n)) {
        println!("column {name} only in right ({ty})");
        problems += 1;
    }
    if left.num_rows() != right.num_rows() {
        println!("rows: left {}, right {}", left.num_rows(), right.num_rows());
        problems += 1;
    }

    // Keys first so sorting groups rows by key, then the rest so ties sort the same on both sides.
    for key in &args.keys {
        if !lt.contains_key(key) || !rt.contains_key(key) {
            bail!("key column {key} is missing from a file");
        }
    }
    let mut columns: Vec<String> = args.keys.clone();
    columns.extend(
        lt.keys().filter(|c| rt.contains_key(*c) && !args.ignore.contains(*c) && !args.keys.contains(c)).cloned(),
    );
    let (mut lrows, mut rrows) = (rows(&left, &columns)?, rows(&right, &columns)?);
    lrows.sort();
    rrows.sort();

    let mut per_column: BTreeMap<&str, usize> = BTreeMap::new();
    let mut shown = 0;
    for (l, r) in lrows.iter().zip(&rrows) {
        for (i, (a, b)) in l.iter().zip(r).enumerate() {
            if a == b {
                continue;
            }
            *per_column.entry(&columns[i]).or_default() += 1;
            if shown < SHOW {
                let key: Vec<&str> = l[..args.keys.len()].iter().map(String::as_str).collect();
                println!("[{}] {}: left {a:?}, right {b:?}", key.join(", "), columns[i]);
                shown += 1;
            }
        }
    }
    for (column, n) in &per_column {
        println!("column {column}: {n} values differ");
        problems += n;
    }
    Ok(problems)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e:#}\nusage: parquet_diff A.parquet B.parquet [--key a,b] [--ignore c,d]");
            return ExitCode::from(2);
        }
    };
    match compare(&args) {
        Ok(0) => {
            println!("same: {} and {}", args.left, args.right);
            ExitCode::SUCCESS
        }
        Ok(_) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
