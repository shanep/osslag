//! A table of nullable string columns, for the package files whose column names depend on the
//! release (`bookworm_version`, `trixie_upstream_version`, ...).

use std::sync::Arc;

use anyhow::{Result, bail};
use arrow::array::{ArrayRef, LargeStringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};

use crate::io::columns::{PANDAS_INDEX_COLUMN, strings};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StringTable {
    names: Vec<String>,
    columns: Vec<Vec<Option<String>>>,
    rows: usize,
}

impl StringTable {
    /// An empty table with the given columns.
    pub fn with_columns<S: AsRef<str>>(names: &[S]) -> Self {
        Self {
            names: names.iter().map(|n| n.as_ref().to_string()).collect(),
            columns: vec![Vec::new(); names.len()],
            rows: 0,
        }
    }

    /// Every column of a batch as strings, leaving out pandas' saved index.
    pub fn from_batch(batch: &RecordBatch) -> Result<Self> {
        let mut table = Self { rows: batch.num_rows(), ..Self::default() };
        for field in batch.schema().fields() {
            if field.name() == PANDAS_INDEX_COLUMN {
                continue;
            }
            table.names.push(field.name().clone());
            table.columns.push(strings(batch, field.name())?);
        }
        Ok(table)
    }

    pub fn to_batch(&self) -> Result<RecordBatch> {
        let fields: Vec<Field> = self.names.iter().map(|n| Field::new(n, DataType::LargeUtf8, true)).collect();
        let arrays: Vec<ArrayRef> = self
            .columns
            .iter()
            .map(|c| Arc::new(c.iter().map(|v| v.as_deref()).collect::<LargeStringArray>()) as ArrayRef)
            .collect();
        let options = arrow::array::RecordBatchOptions::new().with_row_count(Some(self.rows));
        Ok(RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), arrays, &options)?)
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    pub fn column(&self, name: &str) -> Option<&[Option<String>]> {
        self.index(name).map(|i| self.columns[i].as_slice())
    }

    /// The column, or an error naming the columns that are there.
    pub fn require(&self, name: &str, context: &str) -> Result<&[Option<String>]> {
        match self.column(name) {
            Some(c) => Ok(c),
            None => bail!("{context}: missing required columns [{name:?}]. Available columns: {:?}", self.names),
        }
    }

    /// Add a column, or replace one with the same name in place.
    pub fn set_column(&mut self, name: &str, values: Vec<Option<String>>) {
        assert_eq!(values.len(), self.rows, "column {name} length");
        match self.index(name) {
            Some(i) => self.columns[i] = values,
            None => {
                self.names.push(name.to_string());
                self.columns.push(values);
            }
        }
    }

    pub fn rename(&mut self, from: &str, to: &str) {
        if let Some(i) = self.index(from) {
            self.names[i] = to.to_string();
        }
    }

    pub fn drop_columns<S: AsRef<str>>(&mut self, drop: &[S]) {
        let mut i = 0;
        while i < self.names.len() {
            if drop.iter().any(|d| d.as_ref() == self.names[i]) {
                self.names.remove(i);
                self.columns.remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Append a row, given in column order.
    pub fn push_row(&mut self, row: Vec<Option<String>>) {
        assert_eq!(row.len(), self.names.len(), "row width");
        for (col, v) in self.columns.iter_mut().zip(row) {
            col.push(v);
        }
        self.rows += 1;
    }

    pub fn row(&self, i: usize) -> Vec<Option<String>> {
        self.columns.iter().map(|c| c[i].clone()).collect()
    }

    /// Keep the rows where `keep` is true.
    pub fn filter(&self, keep: &[bool]) -> Self {
        let columns = self
            .columns
            .iter()
            .map(|c| c.iter().zip(keep).filter(|(_, k)| **k).map(|(v, _)| v.clone()).collect())
            .collect();
        Self { names: self.names.clone(), columns, rows: keep.iter().filter(|k| **k).count() }
    }

    /// Keep only the first n rows.
    pub fn head(&self, n: usize) -> Self {
        let keep: Vec<bool> = (0..self.rows).map(|i| i < n).collect();
        self.filter(&keep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut t = StringTable::with_columns(&["source", "homepage"]);
        t.push_row(vec![Some("a".into()), None]);
        t.push_row(vec![Some("b".into()), Some("https://github.com/x/y".into())]);
        let back = StringTable::from_batch(&t.to_batch().unwrap()).unwrap();
        assert_eq!(back, t);
        assert_eq!(t.filter(&[false, true]).column("source").unwrap(), &[Some("b".to_string())]);
    }
}
