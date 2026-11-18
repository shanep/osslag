//! Typed access to columns of a record batch, tolerant of the types pandas may have used.

use anyhow::{Result, anyhow, bail};
use arrow::array::{Array, ArrayRef, AsArray, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type, Int64Type, TimeUnit, TimestampMicrosecondType};
use chrono::DateTime;

/// Column names the pandas index sometimes leaves behind. Readers skip them.
pub const PANDAS_INDEX_COLUMN: &str = "__index_level_0__";

pub fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    batch.column_by_name(name).ok_or_else(|| anyhow!("missing column '{name}'"))
}

pub fn has_column(batch: &RecordBatch, name: &str) -> bool {
    batch.column_by_name(name).is_some()
}

/// A string column as owned optional strings.
pub fn strings(batch: &RecordBatch, name: &str) -> Result<Vec<Option<String>>> {
    let col = cast(column(batch, name)?, &DataType::LargeUtf8)?;
    let arr = col.as_string::<i64>();
    Ok((0..arr.len()).map(|i| arr.is_valid(i).then(|| arr.value(i).to_string())).collect())
}

/// A list-of-strings column (for example commit `files` or `tags`).
pub fn string_lists(batch: &RecordBatch, name: &str) -> Result<Vec<Option<Vec<String>>>> {
    let col = column(batch, name)?;
    let target =
        DataType::LargeList(std::sync::Arc::new(arrow::datatypes::Field::new_list_field(DataType::LargeUtf8, true)));
    let col = cast(col, &target)?;
    let lists = col.as_list::<i64>();
    Ok((0..lists.len())
        .map(|i| {
            lists.is_valid(i).then(|| {
                let inner = lists.value(i);
                let inner = inner.as_string::<i64>();
                (0..inner.len()).filter(|&j| inner.is_valid(j)).map(|j| inner.value(j).to_string()).collect()
            })
        })
        .collect())
}

/// A boolean column. Nulls come back as `None`.
pub fn bools(batch: &RecordBatch, name: &str) -> Result<Vec<Option<bool>>> {
    let col = column(batch, name)?;
    let arr = match col.data_type() {
        DataType::Boolean => col.clone(),
        _ => cast(col, &DataType::Boolean)?,
    };
    let arr = arr.as_boolean();
    Ok((0..arr.len()).map(|i| arr.is_valid(i).then(|| arr.value(i))).collect())
}

/// A numeric column as `f64`, the way pandas sees an integer column with gaps.
pub fn floats(batch: &RecordBatch, name: &str) -> Result<Vec<Option<f64>>> {
    let col = cast(column(batch, name)?, &DataType::Float64)?;
    let arr = col.as_primitive::<Float64Type>();
    Ok((0..arr.len()).map(|i| arr.is_valid(i).then(|| arr.value(i))).collect())
}

/// An integer column. Floats must be whole numbers.
pub fn ints(batch: &RecordBatch, name: &str) -> Result<Vec<Option<i64>>> {
    let col = column(batch, name)?;
    if matches!(col.data_type(), DataType::Float32 | DataType::Float64) {
        return floats(batch, name)?
            .into_iter()
            .map(|v| match v {
                Some(f) if f.fract() == 0.0 => Ok(Some(f as i64)),
                Some(f) if f.is_nan() => Ok(None),
                Some(f) => bail!("column '{name}' holds a non-integer value {f}"),
                None => Ok(None),
            })
            .collect();
    }
    let col = cast(col, &DataType::Int64)?;
    let arr = col.as_primitive::<Int64Type>();
    Ok((0..arr.len()).map(|i| arr.is_valid(i).then(|| arr.value(i))).collect())
}

/// A datetime column as microseconds since the Unix epoch in UTC.
///
/// Accepts timestamp columns of any unit (finer units are truncated toward negative infinity,
/// as pandas' `to_pydatetime` does) and string columns holding ISO-8601 or RFC 3339 dates.
/// Naive timestamps and strings without an offset are read as UTC, matching `pd.to_datetime(utc=True)`.
pub fn timestamps_us(batch: &RecordBatch, name: &str) -> Result<Vec<Option<i64>>> {
    let col = column(batch, name)?;
    match col.data_type() {
        DataType::Timestamp(unit, _) => {
            let raw: Vec<Option<i64>> = match unit {
                TimeUnit::Second => prim::<arrow::datatypes::TimestampSecondType>(col),
                TimeUnit::Millisecond => prim::<arrow::datatypes::TimestampMillisecondType>(col),
                TimeUnit::Microsecond => prim::<TimestampMicrosecondType>(col),
                TimeUnit::Nanosecond => prim::<arrow::datatypes::TimestampNanosecondType>(col),
            };
            Ok(raw
                .into_iter()
                .map(|v| {
                    v.map(|v| match unit {
                        TimeUnit::Second => v * 1_000_000,
                        TimeUnit::Millisecond => v * 1_000,
                        TimeUnit::Microsecond => v,
                        TimeUnit::Nanosecond => v.div_euclid(1_000),
                    })
                })
                .collect())
        }
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            strings(batch, name)?.into_iter().map(|v| v.as_deref().map(parse_datetime_us).transpose()).collect()
        }
        DataType::Null => Ok(vec![None; col.len()]),
        other => bail!("column '{name}' has type {other}, expected a timestamp or string"),
    }
}

fn prim<T: arrow::datatypes::ArrowPrimitiveType<Native = i64>>(col: &ArrayRef) -> Vec<Option<i64>> {
    let arr = col.as_primitive::<T>();
    (0..arr.len()).map(|i| arr.is_valid(i).then(|| arr.value(i))).collect()
}

/// Parse an ISO-8601 date-time string to microseconds since the epoch in UTC.
pub fn parse_datetime_us(s: &str) -> Result<i64> {
    let s = s.trim();
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.timestamp_micros());
    }
    for fmt in ["%Y-%m-%d %H:%M:%S%.f%:z", "%Y-%m-%dT%H:%M:%S%.f%z", "%Y-%m-%d %H:%M:%S%.f%z"] {
        if let Ok(dt) = DateTime::parse_from_str(s, fmt) {
            return Ok(dt.timestamp_micros());
        }
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(dt.and_utc().timestamp_micros());
        }
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(d.and_hms_opt(0, 0, 0).expect("midnight").and_utc().timestamp_micros());
    }
    bail!("cannot parse '{s}' as a date-time")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pandas_and_github_dates() {
        let expect = 1_700_000_000_000_000;
        assert_eq!(parse_datetime_us("2023-11-14T22:13:20+00:00").unwrap(), expect);
        assert_eq!(parse_datetime_us("2023-11-14T22:13:20Z").unwrap(), expect);
        assert_eq!(parse_datetime_us("2023-11-14 22:13:20+00:00").unwrap(), expect);
        assert_eq!(parse_datetime_us("2023-11-14 15:13:20-07:00").unwrap(), expect);
        assert_eq!(parse_datetime_us("2023-11-14T22:13:20").unwrap(), expect);
        assert!(parse_datetime_us("yesterday").is_err());
    }
}
