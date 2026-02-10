//! Lightweight GGUF file reader (v2 and v3).
//!
//! Parses the GGUF header, metadata key-value pairs, and tensor index to
//! provide random-access to tensor data. Uses `Read + Seek` generics so that
//! the same code works with `BufReader<File>` on native and `Cursor<&[u8]>`
//! on WASM.

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as little-endian u32: G(0x47) G(0x47) U(0x55) F(0x46)
const ALIGNMENT: u64 = 32;

/// GGML data type codes used in GGUF tensor descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDtype {
    /// 32-bit float (4 bytes per element).
    F32,
    /// 16-bit float (2 bytes per element).
    F16,
    /// 4-bit quantization, block size 32 (18 bytes per block).
    Q4_0,
}

impl GgmlDtype {
    fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            other => bail!("Unsupported GGML dtype code: {other}"),
        }
    }

    /// Byte size for a given number of elements.
    pub fn byte_size(&self, num_elements: u64) -> u64 {
        match self {
            Self::F32 => num_elements * 4,
            Self::F16 => num_elements * 2,
            Self::Q4_0 => {
                // 18 bytes per block of 32 elements
                let num_blocks = num_elements / 32;
                num_blocks * 18
            }
        }
    }
}

/// Metadata for a single tensor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name as stored in the GGUF file.
    pub name: String,
    dimensions: Vec<u64>,
    dtype: GgmlDtype,
    /// Byte offset relative to the start of the data section.
    offset: u64,
}

impl GgufTensorInfo {
    /// Tensor shape as a slice of dimension sizes.
    pub fn shape(&self) -> &[u64] {
        &self.dimensions
    }

    /// The GGML data type of this tensor.
    pub fn dtype(&self) -> GgmlDtype {
        self.dtype
    }

    /// Total number of elements across all dimensions.
    pub fn num_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    /// Total byte size of the tensor data.
    pub fn byte_size(&self) -> u64 {
        self.dtype.byte_size(self.num_elements())
    }
}

/// A reader for GGUF v3 files.
///
/// Parses the header and tensor index on construction, then provides
/// random-access to individual tensor data via [`tensor_data`](Self::tensor_data).
pub struct GgufReader<R: Read + Seek> {
    reader: R,
    version: u32,
    tensor_count: u64,
    tensors: HashMap<String, GgufTensorInfo>,
    data_section_offset: u64,
}

impl GgufReader<Cursor<&[u8]>> {
    /// Open a GGUF file from an in-memory byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<GgufReader<Cursor<&[u8]>>> {
        GgufReader::open(Cursor::new(data))
    }
}

impl<R: Read + Seek> GgufReader<R> {
    /// Parse a GGUF v3 file from the given reader.
    pub fn open(mut reader: R) -> Result<Self> {
        // Magic
        let magic = reader
            .read_u32::<LittleEndian>()
            .context("Failed to read GGUF magic")?;
        if magic != GGUF_MAGIC {
            bail!("Invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})");
        }

        // Version (support v2 and v3)
        let version = reader
            .read_u32::<LittleEndian>()
            .context("Failed to read GGUF version")?;
        if version != 2 && version != 3 {
            bail!("Unsupported GGUF version: {version} (expected 2 or 3)");
        }

        // Counts
        let tensor_count = reader
            .read_u64::<LittleEndian>()
            .context("Failed to read tensor count")?;
        let metadata_kv_count = reader
            .read_u64::<LittleEndian>()
            .context("Failed to read metadata KV count")?;

        // Skip metadata key-value pairs
        for i in 0..metadata_kv_count {
            let _key = read_gguf_string(&mut reader)
                .with_context(|| format!("Failed to read metadata key {i}"))?;
            let value_type = reader
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read metadata value type {i}"))?;
            skip_gguf_value(&mut reader, value_type)
                .with_context(|| format!("Failed to skip metadata value {i}"))?;
        }

        // Parse tensor index
        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = read_gguf_string(&mut reader)
                .with_context(|| format!("Failed to read tensor name {i}"))?;
            let ndims = reader
                .read_u32::<LittleEndian>()
                .with_context(|| format!("Failed to read ndims for tensor {i}"))?;
            let mut dimensions = Vec::with_capacity(ndims as usize);
            for d in 0..ndims {
                dimensions.push(
                    reader
                        .read_u64::<LittleEndian>()
                        .with_context(|| format!("Failed to read dim {d} for tensor {i}"))?,
                );
            }
            let dtype = GgmlDtype::from_u32(
                reader
                    .read_u32::<LittleEndian>()
                    .with_context(|| format!("Failed to read dtype for tensor {i}"))?,
            )?;
            let offset = reader
                .read_u64::<LittleEndian>()
                .with_context(|| format!("Failed to read offset for tensor {i}"))?;

            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    dimensions,
                    dtype,
                    offset,
                },
            );
        }

        // Data section starts at next 32-byte boundary
        let current_pos = reader.stream_position()?;
        let data_section_offset = align_up(current_pos, ALIGNMENT);

        Ok(Self {
            reader,
            version,
            tensor_count,
            tensors,
            data_section_offset,
        })
    }

    /// GGUF format version (expected: 3).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Number of tensors in the file.
    pub fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    /// Look up metadata for a tensor by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Read raw tensor data bytes from the file.
    pub fn tensor_data(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("Tensor '{name}' not found in GGUF"))?
            .clone();
        let byte_size = info.byte_size() as usize;
        let abs_offset = self.data_section_offset + info.offset;
        self.reader.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; byte_size];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).context("Invalid UTF-8 in GGUF string")
}

fn skip_gguf_value<R: Read + Seek>(reader: &mut R, value_type: u32) -> Result<()> {
    match value_type {
        0 => {
            reader.read_u8()?;
        } // u8
        1 => {
            reader.read_i8()?;
        } // i8
        2 => {
            reader.seek(SeekFrom::Current(2))?;
        } // u16
        3 => {
            reader.seek(SeekFrom::Current(2))?;
        } // i16
        4 => {
            reader.seek(SeekFrom::Current(4))?;
        } // u32
        5 => {
            reader.seek(SeekFrom::Current(4))?;
        } // i32
        6 => {
            reader.seek(SeekFrom::Current(4))?;
        } // f32
        7 => {
            reader.read_u8()?;
        } // bool (1 byte)
        8 => {
            let _ = read_gguf_string(reader)?;
        } // string
        9 => {
            // array
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;
            for _ in 0..count {
                skip_gguf_value(reader, elem_type)?;
            }
        }
        10 => {
            reader.seek(SeekFrom::Current(8))?;
        } // u64
        11 => {
            reader.seek(SeekFrom::Current(8))?;
        } // i64
        12 => {
            reader.seek(SeekFrom::Current(8))?;
        } // f64
        other => bail!("Unknown GGUF metadata value type: {other}"),
    }
    Ok(())
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}
