//! Parquet reading and writing shared by the pipeline and the metrics.
//!
//! Readers accept whatever pandas has written over the years (string or large_string,
//! timestamps in s/ms/us/ns, ISO-8601 strings for dates). Writers emit what pandas 3 writes
//! today (large_string, timestamp[us, UTC]) so Python notebooks read the files unchanged.

pub mod columns;
pub mod table;

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

/// Read a whole parquet file into one batch.
pub fn read_parquet(path: &Path) -> Result<RecordBatch> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).with_context(|| format!("reading {}", path.display()))?;
    let schema = builder.schema().clone();
    let batches = builder.build()?.collect::<Result<Vec<_>, _>>()?;
    Ok(concat_batches(&schema, &batches)?)
}

/// Write one batch to a parquet file, replacing it if it exists.
pub fn write_parquet(path: &Path, batch: &RecordBatch) -> Result<()> {
    write_batches(path, batch.schema(), std::slice::from_ref(batch))
}

/// Write several batches with the same schema to one parquet file.
pub fn write_batches(path: &Path, schema: SchemaRef, batches: &[RecordBatch]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Write to a temporary file first so a crash never leaves a truncated checkpoint behind.
    let tmp = path.with_extension("parquet.tmp");
    let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let props = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::default())).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}
