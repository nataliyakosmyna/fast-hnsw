#![doc = include_str!("../README.md")]

use crc32fast::Hasher;
use memmap2::{Mmap, MmapOptions};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const SNAPSHOT_MAGIC: &[u8; 8] = b"FHQVEC\0\0";
const SNAPSHOT_VERSION: u16 = 1;
const HEADER_BYTES: usize = 80;
const CHECKSUM_BYTES: usize = size_of::<u32>();
const SCALE_SLOT_FLAG: u32 = 1;
const BLOCK_CANDIDATES: [usize; 7] = [512, 256, 128, 64, 32, 16, 8];
const LOADING_FACTOR: [f32; 8] = [1.0, 1.6, 2.0, 2.4, 2.7, 3.0, 3.2, 3.4];
// Lloyd-Max reconstruction levels for a unit Gaussian.
const LLOYD_MAX_HALF: [&[f32]; 4] = [
    &[0.7979],
    &[0.4528, 1.5104],
    &[0.2451, 0.7560, 1.3439, 2.1520],
    &[
        0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326,
    ],
];

/// Errors returned while encoding, opening, or scoring a quantized sidecar.
#[derive(Debug)]
pub enum QuantizedError {
    Io(std::io::Error),
    InvalidBits(u8),
    ZeroDimensions,
    DimensionMismatch { expected: usize, actual: usize },
    NonFinite,
    ZeroNorm,
    RowOutOfBounds { row: usize, rows: usize },
    IncompatibleQuery(String),
    IncompatibleIndex(String),
    InvalidSnapshot(String),
    /// An error raised by the underlying HNSW index.
    #[cfg(feature = "hnsw")]
    Hnsw(fast_hnsw::Error),
}

#[cfg(feature = "hnsw")]
impl From<fast_hnsw::Error> for QuantizedError {
    fn from(error: fast_hnsw::Error) -> Self {
        Self::Hnsw(error)
    }
}

impl Display for QuantizedError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => Display::fmt(error, formatter),
            Self::InvalidBits(bits) => write!(formatter, "unsupported quantization width {bits}; expected 2, 3, or 4 bits"),
            Self::ZeroDimensions => formatter.write_str("vector dimensions must be non-zero"),
            Self::DimensionMismatch { expected, actual } => {
                write!(formatter, "vector dimension mismatch: expected {expected}, got {actual}")
            }
            Self::NonFinite => formatter.write_str("vectors must contain only finite values"),
            Self::ZeroNorm => formatter.write_str("cosine vectors must have non-zero norm"),
            Self::RowOutOfBounds { row, rows } => {
                write!(formatter, "quantized row {row} is out of bounds for {rows} rows")
            }
            Self::IncompatibleQuery(message) => {
                write!(formatter, "incompatible prepared query: {message}")
            }
            Self::IncompatibleIndex(message) => write!(formatter, "incompatible HNSW index: {message}"),
            Self::InvalidSnapshot(message) => write!(formatter, "invalid quantized snapshot: {message}"),
            #[cfg(feature = "hnsw")]
            Self::Hnsw(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for QuantizedError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            #[cfg(feature = "hnsw")]
            Self::Hnsw(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for QuantizedError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Similarity represented by the quantized rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedMetric {
    Cosine,
    DotProduct,
}

impl QuantizedMetric {
    fn tag(self) -> u8 {
        match self {
            Self::Cosine => 0,
            Self::DotProduct => 1,
        }
    }

    fn from_tag(tag: u8) -> Result<Self, QuantizedError> {
        match tag {
            0 => Ok(Self::Cosine),
            1 => Ok(Self::DotProduct),
            _ => Err(invalid(format!("unknown metric tag {tag}"))),
        }
    }
}

/// Stable settings persisted in every quantized sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedConfig {
    /// Quantized bits per padded dimension. Version 1 supports 2–4 bits.
    pub bits: u8,
    /// Seed for the versioned Rademacher-sign generator.
    pub seed: u64,
}

impl Default for QuantizedConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            seed: 0x4648_5156,
        }
    }
}

impl QuantizedConfig {
    pub const MIN_BITS: u8 = 2;
    pub const MAX_BITS: u8 = 4;

    pub fn with_bits(bits: u8) -> Result<Self, QuantizedError> {
        Self::new(bits, Self::default().seed)
    }

    pub fn new(bits: u8, seed: u64) -> Result<Self, QuantizedError> {
        let config = Self { bits, seed };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(self) -> Result<(), QuantizedError> {
        if !(Self::MIN_BITS..=Self::MAX_BITS).contains(&self.bits) {
            return Err(QuantizedError::InvalidBits(self.bits));
        }
        Ok(())
    }
}

/// Validated metadata for a mapped sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedMetadata {
    pub version: u16,
    pub dimensions: usize,
    pub padded_dimensions: usize,
    pub rows: usize,
    pub metric: QuantizedMetric,
    pub config: QuantizedConfig,
    pub block_size: usize,
    pub row_stride: usize,
}

#[derive(Debug)]
struct CodecParameters {
    dimensions: usize,
    padded_dimensions: usize,
    block_size: usize,
    bits: u8,
    config: QuantizedConfig,
    signs: Vec<f32>,
    /// Reconstruction levels padded to the 4-bit code space so a masked code
    /// always indexes in bounds; only the first `1 << bits` entries are levels.
    level_table: [f32; 16],
    boundaries: Vec<f32>,
}

impl CodecParameters {
    fn new(dimensions: usize, config: QuantizedConfig) -> Result<Self, QuantizedError> {
        if dimensions == 0 {
            return Err(QuantizedError::ZeroDimensions);
        }
        config.validate()?;
        let (block_size, padded_dimensions) = blocking(dimensions);
        let signs = stable_signs(padded_dimensions, config.seed);
        let levels = levels(config.bits, padded_dimensions);
        let boundaries = levels
            .windows(2)
            .map(|window| 0.5 * (window[0] + window[1]))
            .collect();
        let mut level_table = [0.0; 16];
        level_table[..levels.len()].copy_from_slice(&levels);
        Ok(Self {
            dimensions,
            padded_dimensions,
            block_size,
            bits: config.bits,
            config,
            signs,
            level_table,
            boundaries,
        })
    }

    fn row_stride(&self) -> usize {
        (self.padded_dimensions * usize::from(self.bits)).div_ceil(8)
    }

    fn prepare_query(
        &self,
        metric: QuantizedMetric,
        query: &[f32],
    ) -> Result<PreparedQuantizedQuery, QuantizedError> {
        validate_vector(query, self.dimensions)?;
        let mut rotated = vec![0.0; self.padded_dimensions];
        rotated[..self.dimensions].copy_from_slice(query);
        if metric == QuantizedMetric::Cosine {
            normalize(&mut rotated[..self.dimensions])?;
        }
        rotate(&mut rotated, &self.signs, self.block_size);
        Ok(PreparedQuantizedQuery {
            rotated,
            identity: CodecIdentity {
                metric,
                dimensions: self.dimensions,
                padded_dimensions: self.padded_dimensions,
                block_size: self.block_size,
                config: self.config,
            },
        })
    }

    fn encode(
        &self,
        metric: QuantizedMetric,
        vector: &[f32],
    ) -> Result<(Vec<u8>, f32), QuantizedError> {
        validate_vector(vector, self.dimensions)?;
        let norm = l2_norm(vector);
        if metric == QuantizedMetric::Cosine && norm == 0.0 {
            return Err(QuantizedError::ZeroNorm);
        }
        let stored_norm = if metric == QuantizedMetric::Cosine {
            1.0
        } else {
            norm
        };
        let mut rotated = vec![0.0; self.padded_dimensions];
        if norm > f32::EPSILON {
            for (output, input) in rotated[..self.dimensions].iter_mut().zip(vector) {
                *output = *input / norm;
            }
        }
        rotate(&mut rotated, &self.signs, self.block_size);

        let mut codes = vec![0; self.row_stride()];
        let mut reconstruction_norm_squared = 0.0;
        for (dimension, value) in rotated.into_iter().enumerate() {
            let code = self
                .boundaries
                .partition_point(|boundary| *boundary < value);
            pack_code(&mut codes, self.bits, dimension, code as u8);
            let level = self.level_table[code];
            reconstruction_norm_squared = level.mul_add(level, reconstruction_norm_squared);
        }
        if reconstruction_norm_squared <= f32::EPSILON {
            return Err(invalid("quantized row reconstructs to zero"));
        }
        Ok((codes, stored_norm / reconstruction_norm_squared.sqrt()))
    }
}

/// Query state prepared once and reused for every candidate visited by HNSW.
#[derive(Debug, Clone)]
pub struct PreparedQuantizedQuery {
    rotated: Vec<f32>,
    identity: CodecIdentity,
}

impl PreparedQuantizedQuery {
    pub fn padded_dimensions(&self) -> usize {
        self.rotated.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CodecIdentity {
    metric: QuantizedMetric,
    dimensions: usize,
    padded_dimensions: usize,
    block_size: usize,
    config: QuantizedConfig,
}

/// Read-only mmap view over a versioned low-bit vector sidecar.
pub struct MappedQuantizedVectors {
    image: Mmap,
    metadata: QuantizedMetadata,
    parameters: CodecParameters,
    codes_offset: usize,
    scales_offset: usize,
    checksum_offset: usize,
    checksum_verified: bool,
}

impl MappedQuantizedVectors {
    /// Writes a deterministic sidecar to `path`.
    ///
    /// This method does not replace files atomically. Applications that need an
    /// atomic lifecycle should write to a temporary sibling and rename it only
    /// after opening it with [`Self::open_mmap_verified`].
    pub fn save<V: AsRef<[f32]>>(
        path: impl AsRef<Path>,
        dimensions: usize,
        metric: QuantizedMetric,
        config: QuantizedConfig,
        vectors: &[V],
    ) -> Result<(), QuantizedError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        Self::write_to(&mut writer, dimensions, metric, config, vectors)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Writes one complete, self-contained sidecar at the writer's current position.
    ///
    /// Offsets inside the image are relative to the start of this sidecar, so
    /// the returned byte count can later be passed to [`Self::open_mmap_region`].
    pub fn write_to<W: Write, V: AsRef<[f32]>>(
        writer: &mut W,
        dimensions: usize,
        metric: QuantizedMetric,
        config: QuantizedConfig,
        vectors: &[V],
    ) -> Result<usize, QuantizedError> {
        let parameters = CodecParameters::new(dimensions, config)?;
        let layout = SnapshotLayout::new(vectors.len(), parameters.row_stride())?;
        let mut checksum = Hasher::new();
        let header = build_header(
            dimensions,
            vectors.len(),
            metric,
            config,
            &parameters,
            layout,
        )?;
        write_hashed(writer, &mut checksum, &header)?;

        let mut scales = Vec::with_capacity(vectors.len());
        for vector in vectors {
            let (codes, scale) = parameters.encode(metric, vector.as_ref())?;
            write_hashed(writer, &mut checksum, &codes)?;
            scales.push(scale);
        }
        for scale in scales {
            write_hashed(writer, &mut checksum, &scale.to_le_bytes())?;
        }
        writer.write_all(&checksum.finalize().to_le_bytes())?;
        Ok(layout.file_bytes)
    }

    /// Opens a structurally validated mapping without scanning all packed rows.
    ///
    /// # Safety
    ///
    /// The file must not be modified or truncated for the mapping's lifetime.
    pub unsafe fn open_mmap(path: impl AsRef<Path>) -> Result<Self, QuantizedError> {
        // SAFETY: the caller accepts the documented immutable-file contract.
        unsafe { Self::open_internal(path.as_ref(), 0, None, false) }
    }

    /// Opens a mapping and verifies its whole-content CRC before returning.
    ///
    /// # Safety
    ///
    /// The file must not be modified or truncated for the mapping's lifetime.
    pub unsafe fn open_mmap_verified(path: impl AsRef<Path>) -> Result<Self, QuantizedError> {
        // SAFETY: the caller accepts the documented immutable-file contract.
        unsafe { Self::open_internal(path.as_ref(), 0, None, true) }
    }

    /// Opens one sidecar embedded within a larger immutable file.
    ///
    /// `offset` and `length` identify the exact image written by
    /// [`Self::write_to`]. Set `verify_checksum` for whole-region CRC validation.
    ///
    /// # Safety
    ///
    /// The containing file must not be modified or truncated for the mapping's
    /// lifetime, including bytes outside the selected region.
    pub unsafe fn open_mmap_region(
        path: impl AsRef<Path>,
        offset: u64,
        length: usize,
        verify_checksum: bool,
    ) -> Result<Self, QuantizedError> {
        // SAFETY: the caller accepts the documented immutable-file contract.
        unsafe { Self::open_internal(path.as_ref(), offset, Some(length), verify_checksum) }
    }

    unsafe fn open_internal(
        path: &Path,
        offset: u64,
        mapped_length: Option<usize>,
        verify_checksum: bool,
    ) -> Result<Self, QuantizedError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if offset > file_len {
            return Err(invalid("mapping offset exceeds file length"));
        }
        let remaining = file_len - offset;
        let region_len = match mapped_length {
            Some(length) => length,
            None => usize::try_from(remaining)
                .map_err(|_| invalid("file length exceeds addressable memory"))?,
        };
        let region_end = offset
            .checked_add(
                u64::try_from(region_len)
                    .map_err(|_| invalid("mapping length exceeds u64"))?,
            )
            .ok_or_else(|| invalid("mapping region overflows its offset"))?;
        if region_end > file_len {
            return Err(invalid("mapping region exceeds file length"));
        }
        if region_len < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(invalid("snapshot is shorter than its header"));
        }
        // SAFETY: the public caller guarantees that the file remains immutable.
        let image = unsafe { MmapOptions::new().offset(offset).len(region_len).map(&file)? };
        let parsed = ParsedHeader::parse(&image)?;
        let parameters = CodecParameters::new(parsed.dimensions, parsed.config)?;
        if parsed.padded_dimensions != parameters.padded_dimensions
            || parsed.block_size != parameters.block_size
            || parsed.row_stride != parameters.row_stride()
        {
            return Err(invalid("codec parameters do not match the format version"));
        }
        let layout = SnapshotLayout::new(parsed.rows, parsed.row_stride)?;
        if parsed.codes_offset != layout.codes_offset
            || parsed.scales_offset != layout.scales_offset
            || parsed.checksum_offset != layout.checksum_offset
        {
            return Err(invalid("section offsets are inconsistent"));
        }
        if image.len() != layout.file_bytes {
            return Err(invalid("file length does not match its metadata"));
        }
        for row in 0..parsed.rows {
            let scale = read_scale(&image, layout.scales_offset, row)?;
            if !scale.is_finite() || scale < 0.0 {
                return Err(invalid("row scale is not finite and non-negative"));
            }
        }
        if verify_checksum {
            let expected = read_u32_at(&image, layout.checksum_offset)?;
            let mut checksum = Hasher::new();
            checksum.update(&image[..layout.checksum_offset]);
            if checksum.finalize() != expected {
                return Err(invalid("checksum mismatch"));
            }
        }
        Ok(Self {
            image,
            metadata: QuantizedMetadata {
                version: SNAPSHOT_VERSION,
                dimensions: parsed.dimensions,
                padded_dimensions: parsed.padded_dimensions,
                rows: parsed.rows,
                metric: parsed.metric,
                config: parsed.config,
                block_size: parsed.block_size,
                row_stride: parsed.row_stride,
            },
            parameters,
            codes_offset: layout.codes_offset,
            scales_offset: layout.scales_offset,
            checksum_offset: layout.checksum_offset,
            checksum_verified: verify_checksum,
        })
    }

    pub fn metadata(&self) -> QuantizedMetadata {
        self.metadata
    }

    pub fn len(&self) -> usize {
        self.metadata.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn mapped_bytes(&self) -> usize {
        self.image.len()
    }

    pub fn packed_bytes(&self) -> usize {
        self.checksum_offset - self.codes_offset
    }

    pub fn is_checksum_verified(&self) -> bool {
        self.checksum_verified
    }

    pub fn prepare_query(
        &self,
        query: &[f32],
    ) -> Result<PreparedQuantizedQuery, QuantizedError> {
        self.parameters.prepare_query(self.metadata.metric, query)
    }

    /// Returns approximate similarity: cosine similarity or raw dot product.
    pub fn score_prepared(
        &self,
        query: &PreparedQuantizedQuery,
        row: usize,
    ) -> Result<f32, QuantizedError> {
        if row >= self.len() {
            return Err(QuantizedError::RowOutOfBounds {
                row,
                rows: self.len(),
            });
        }
        self.validate_prepared(query)?;
        Ok(self.score_prepared_validated(query, row))
    }

    pub fn score(&self, query: &[f32], row: usize) -> Result<f32, QuantizedError> {
        let prepared = self.prepare_query(query)?;
        self.score_prepared(&prepared, row)
    }

    #[inline]
    fn score_prepared_validated(&self, query: &PreparedQuantizedQuery, row: usize) -> f32 {
        let start = self.codes_offset + row * self.metadata.row_stride;
        let codes = &self.image[start..start + self.metadata.row_stride];
        let scale = f32::from_le_bytes(
            self.image[self.scales_offset + row * 4..self.scales_offset + row * 4 + 4]
                .try_into()
                .expect("validated four-byte scale slot"),
        );
        debug_assert_eq!(query.rotated.len() % 8, 0);
        let query = query.rotated.as_slice();
        let levels = &self.parameters.level_table;
        let sum = match self.metadata.config.bits {
            2 => accumulate_row_codes::<2>(query, codes, levels),
            3 => accumulate_row_codes::<3>(query, codes, levels),
            4 => accumulate_row_codes::<4>(query, codes, levels),
            _ => unreachable!("bit width was validated while opening"),
        };
        sum * scale
    }

    #[inline]
    fn distance_prepared_validated(&self, query: &PreparedQuantizedQuery, row: usize) -> f32 {
        1.0 - self.score_prepared_validated(query, row)
    }

    fn validate_prepared(&self, query: &PreparedQuantizedQuery) -> Result<(), QuantizedError> {
        let expected = CodecIdentity {
            metric: self.metadata.metric,
            dimensions: self.metadata.dimensions,
            padded_dimensions: self.metadata.padded_dimensions,
            block_size: self.metadata.block_size,
            config: self.metadata.config,
        };
        if query.identity != expected {
            return Err(QuantizedError::IncompatibleQuery(
                "metric, dimensions, or codec settings differ from the mapped sidecar".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "hnsw")]
mod hnsw_integration {
    use super::{MappedQuantizedVectors, PreparedQuantizedQuery, QuantizedError};
    use fast_hnsw::distance::Distance;
    use fast_hnsw::{Hnsw, SearchResult, SearchWorkspace};

    /// Validated pairing of an HNSW graph and its mmap quantized vector rows.
    pub struct QuantizedHnsw<'a, D: Distance> {
        index: &'a Hnsw<D>,
        vectors: &'a MappedQuantizedVectors,
    }

    impl<'a, D: Distance> QuantizedHnsw<'a, D> {
        pub fn new(
            index: &'a Hnsw<D>,
            vectors: &'a MappedQuantizedVectors,
        ) -> Result<Self, QuantizedError> {
            if index.len() != vectors.len() {
                return Err(QuantizedError::IncompatibleIndex(format!(
                    "graph has {} nodes but sidecar has {} rows",
                    index.len(),
                    vectors.len()
                )));
            }
            if let Some(dimensions) = index.dim() {
                if dimensions != vectors.metadata().dimensions {
                    return Err(QuantizedError::IncompatibleIndex(format!(
                        "graph has {dimensions} dimensions but sidecar has {}",
                        vectors.metadata().dimensions
                    )));
                }
            }
            Ok(Self { index, vectors })
        }

        pub fn prepare_query(
            &self,
            query: &[f32],
        ) -> Result<PreparedQuantizedQuery, QuantizedError> {
            self.vectors.prepare_query(query)
        }

        pub fn search(
            &self,
            query: &[f32],
            k: usize,
            ef: usize,
        ) -> Result<Vec<SearchResult>, QuantizedError> {
            let prepared = self.prepare_query(query)?;
            self.search_prepared(&prepared, k, ef)
        }

        pub fn search_prepared(
            &self,
            query: &PreparedQuantizedQuery,
            k: usize,
            ef: usize,
        ) -> Result<Vec<SearchResult>, QuantizedError> {
            self.vectors.validate_prepared(query)?;
            Ok(self.index.search_with_distance(k, ef, |id| {
                self.vectors.distance_prepared_validated(query, id)
            })?)
        }

        pub fn search_prepared_with_workspace(
            &self,
            query: &PreparedQuantizedQuery,
            k: usize,
            ef: usize,
            workspace: &mut SearchWorkspace,
        ) -> Result<Vec<SearchResult>, QuantizedError> {
            self.vectors.validate_prepared(query)?;
            Ok(self.index.search_with_distance_and_workspace(
                k,
                ef,
                |id| self.vectors.distance_prepared_validated(query, id),
                workspace,
            )?)
        }

        pub fn search_filtered<A>(
            &self,
            query: &[f32],
            k: usize,
            ef: usize,
            accepts: A,
        ) -> Result<Vec<SearchResult>, QuantizedError>
        where
            A: Fn(usize) -> bool,
        {
            let prepared = self.prepare_query(query)?;
            Ok(self.index.search_filtered_with_distance(k, ef, |id| {
                self.vectors.distance_prepared_validated(&prepared, id)
            }, accepts)?)
        }

        pub fn search_filtered_prepared_with_workspace<A>(
            &self,
            query: &PreparedQuantizedQuery,
            k: usize,
            ef: usize,
            accepts: A,
            workspace: &mut SearchWorkspace,
        ) -> Result<Vec<SearchResult>, QuantizedError>
        where
            A: Fn(usize) -> bool,
        {
            self.vectors.validate_prepared(query)?;
            Ok(self.index.search_filtered_with_distance_and_workspace(
                k,
                ef,
                |id| self.vectors.distance_prepared_validated(query, id),
                accepts,
                workspace,
            )?)
        }
    }
}

#[cfg(feature = "hnsw")]
pub use hnsw_integration::QuantizedHnsw;

#[derive(Debug, Clone, Copy)]
struct SnapshotLayout {
    codes_offset: usize,
    scales_offset: usize,
    checksum_offset: usize,
    file_bytes: usize,
}

impl SnapshotLayout {
    fn new(rows: usize, row_stride: usize) -> Result<Self, QuantizedError> {
        let codes_offset = HEADER_BYTES;
        let scales_offset = codes_offset
            .checked_add(
                rows.checked_mul(row_stride)
                    .ok_or_else(|| invalid("packed row size overflow"))?,
            )
            .ok_or_else(|| invalid("scale offset overflow"))?;
        let checksum_offset = scales_offset
            .checked_add(
                rows.checked_mul(size_of::<f32>())
                    .ok_or_else(|| invalid("scale section size overflow"))?,
            )
            .ok_or_else(|| invalid("checksum offset overflow"))?;
        let file_bytes = checksum_offset
            .checked_add(CHECKSUM_BYTES)
            .ok_or_else(|| invalid("file size overflow"))?;
        Ok(Self {
            codes_offset,
            scales_offset,
            checksum_offset,
            file_bytes,
        })
    }
}

#[derive(Debug)]
struct ParsedHeader {
    metric: QuantizedMetric,
    dimensions: usize,
    padded_dimensions: usize,
    rows: usize,
    config: QuantizedConfig,
    block_size: usize,
    row_stride: usize,
    codes_offset: usize,
    scales_offset: usize,
    checksum_offset: usize,
}

impl ParsedHeader {
    fn parse(image: &[u8]) -> Result<Self, QuantizedError> {
        let mut offset = 0;
        if take::<8>(image, &mut offset)? != *SNAPSHOT_MAGIC {
            return Err(invalid("bad magic"));
        }
        let version = u16::from_le_bytes(take(image, &mut offset)?);
        if version != SNAPSHOT_VERSION {
            return Err(invalid(format!("unsupported version {version}")));
        }
        let header_bytes = usize::from(u16::from_le_bytes(take(image, &mut offset)?));
        if header_bytes != HEADER_BYTES {
            return Err(invalid("header length does not match the format version"));
        }
        let metric = QuantizedMetric::from_tag(take::<1>(image, &mut offset)?[0])?;
        let bits = take::<1>(image, &mut offset)?[0];
        let reserved_short = u16::from_le_bytes(take(image, &mut offset)?);
        let dimensions = u32::from_le_bytes(take(image, &mut offset)?) as usize;
        let padded_dimensions = u32::from_le_bytes(take(image, &mut offset)?) as usize;
        let block_size = u32::from_le_bytes(take(image, &mut offset)?) as usize;
        let row_stride = u32::from_le_bytes(take(image, &mut offset)?) as usize;
        let rows = to_usize(u64::from_le_bytes(take(image, &mut offset)?))?;
        let seed = u64::from_le_bytes(take(image, &mut offset)?);
        let codes_offset = to_usize(u64::from_le_bytes(take(image, &mut offset)?))?;
        let scales_offset = to_usize(u64::from_le_bytes(take(image, &mut offset)?))?;
        let checksum_offset = to_usize(u64::from_le_bytes(take(image, &mut offset)?))?;
        let flags = u32::from_le_bytes(take(image, &mut offset)?);
        let reserved = u32::from_le_bytes(take(image, &mut offset)?);
        if offset != HEADER_BYTES || reserved_short != 0 || reserved != 0 {
            return Err(invalid("header contains non-zero reserved data"));
        }
        if flags != SCALE_SLOT_FLAG {
            return Err(invalid("unknown snapshot flags"));
        }
        Ok(Self {
            metric,
            dimensions,
            padded_dimensions,
            rows,
            config: QuantizedConfig { bits, seed },
            block_size,
            row_stride,
            codes_offset,
            scales_offset,
            checksum_offset,
        })
    }
}

fn build_header(
    dimensions: usize,
    rows: usize,
    metric: QuantizedMetric,
    config: QuantizedConfig,
    parameters: &CodecParameters,
    layout: SnapshotLayout,
) -> Result<Vec<u8>, QuantizedError> {
    let dimensions = u32::try_from(dimensions).map_err(|_| invalid("dimensions exceed u32"))?;
    let padded_dimensions = u32::try_from(parameters.padded_dimensions)
        .map_err(|_| invalid("padded dimensions exceed u32"))?;
    let block_size = u32::try_from(parameters.block_size)
        .map_err(|_| invalid("block size exceeds u32"))?;
    let row_stride = u32::try_from(parameters.row_stride())
        .map_err(|_| invalid("row stride exceeds u32"))?;
    let rows = u64::try_from(rows).map_err(|_| invalid("row count exceeds u64"))?;
    let mut header = Vec::with_capacity(HEADER_BYTES);
    header.extend_from_slice(SNAPSHOT_MAGIC);
    header.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
    header.extend_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
    header.push(metric.tag());
    header.push(config.bits);
    header.extend_from_slice(&0_u16.to_le_bytes());
    header.extend_from_slice(&dimensions.to_le_bytes());
    header.extend_from_slice(&padded_dimensions.to_le_bytes());
    header.extend_from_slice(&block_size.to_le_bytes());
    header.extend_from_slice(&row_stride.to_le_bytes());
    header.extend_from_slice(&rows.to_le_bytes());
    header.extend_from_slice(&config.seed.to_le_bytes());
    header.extend_from_slice(&(layout.codes_offset as u64).to_le_bytes());
    header.extend_from_slice(&(layout.scales_offset as u64).to_le_bytes());
    header.extend_from_slice(&(layout.checksum_offset as u64).to_le_bytes());
    header.extend_from_slice(&SCALE_SLOT_FLAG.to_le_bytes());
    header.extend_from_slice(&0_u32.to_le_bytes());
    debug_assert_eq!(header.len(), HEADER_BYTES);
    Ok(header)
}

fn invalid(message: impl Into<String>) -> QuantizedError {
    QuantizedError::InvalidSnapshot(message.into())
}

fn validate_vector(vector: &[f32], dimensions: usize) -> Result<(), QuantizedError> {
    if vector.len() != dimensions {
        return Err(QuantizedError::DimensionMismatch {
            expected: dimensions,
            actual: vector.len(),
        });
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(QuantizedError::NonFinite);
    }
    Ok(())
}

fn write_hashed(
    writer: &mut impl Write,
    checksum: &mut Hasher,
    bytes: &[u8],
) -> std::io::Result<()> {
    writer.write_all(bytes)?;
    checksum.update(bytes);
    Ok(())
}

fn read_scale(image: &[u8], offset: usize, row: usize) -> Result<f32, QuantizedError> {
    let start = offset
        .checked_add(
            row.checked_mul(size_of::<f32>())
                .ok_or_else(|| invalid("scale offset overflow"))?,
        )
        .ok_or_else(|| invalid("scale offset overflow"))?;
    Ok(f32::from_bits(read_u32_at(image, start)?))
}

fn read_u32_at(image: &[u8], offset: usize) -> Result<u32, QuantizedError> {
    let bytes = image
        .get(offset..offset + 4)
        .ok_or_else(|| invalid("read exceeds file"))?;
    Ok(u32::from_le_bytes(
        bytes.try_into().expect("four-byte range"),
    ))
}

fn take<const N: usize>(image: &[u8], offset: &mut usize) -> Result<[u8; N], QuantizedError> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| invalid("header offset overflow"))?;
    let bytes = image
        .get(*offset..end)
        .ok_or_else(|| invalid("header is truncated"))?;
    *offset = end;
    Ok(bytes.try_into().expect("fixed-size header range"))
}

fn to_usize(value: u64) -> Result<usize, QuantizedError> {
    usize::try_from(value).map_err(|_| invalid("value exceeds usize"))
}

fn blocking(dimensions: usize) -> (usize, usize) {
    for block in BLOCK_CANDIDATES {
        if dimensions % block == 0 {
            return (block, dimensions);
        }
    }
    let block = 8;
    (block, dimensions.div_ceil(block) * block)
}

fn levels(bits: u8, dimensions: usize) -> Vec<f32> {
    let sigma = 1.0 / (dimensions as f32).sqrt();
    let half = LLOYD_MAX_HALF[usize::from(bits) - 1];
    let mut values = half
        .iter()
        .rev()
        .map(|value| -*value * sigma)
        .collect::<Vec<_>>();
    values.extend(half.iter().map(|value| *value * sigma));
    if values.len() == 1_usize << bits {
        return values;
    }
    let count = 1_usize << bits;
    let radius = LOADING_FACTOR[usize::from(bits) - 1] * sigma;
    let step = 2.0 * radius / count as f32;
    (0..count)
        .map(|index| -radius + (index as f32 + 0.5) * step)
        .collect()
}

fn stable_signs(dimensions: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..dimensions)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            mixed ^= mixed >> 31;
            if mixed & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

fn l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn normalize(vector: &mut [f32]) -> Result<(), QuantizedError> {
    let norm = l2_norm(vector);
    if norm == 0.0 {
        return Err(QuantizedError::ZeroNorm);
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

fn rotate(values: &mut [f32], signs: &[f32], block_size: usize) {
    for (value, sign) in values.iter_mut().zip(signs) {
        *value *= sign;
    }
    for block in values.chunks_mut(block_size) {
        fwht(block);
    }
}

fn fwht(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut half = 1;
    while half < values.len() {
        for start in (0..values.len()).step_by(half * 2) {
            for offset in 0..half {
                let left = values[start + offset];
                let right = values[start + offset + half];
                values[start + offset] = left + right;
                values[start + offset + half] = left - right;
            }
        }
        half *= 2;
    }
    let scale = 1.0 / (values.len() as f32).sqrt();
    for value in values {
        *value *= scale;
    }
}

fn pack_code(output: &mut [u8], bits: u8, index: usize, value: u8) {
    let bit_offset = index * usize::from(bits);
    let byte = bit_offset / 8;
    let shift = bit_offset % 8;
    let shifted = u16::from(value) << shift;
    output[byte] |= shifted as u8;
    if shift + usize::from(bits) > 8 {
        output[byte + 1] |= (shifted >> 8) as u8;
    }
}

/// Each `BITS`-byte chunk of `codes` packs eight consecutive dimensions.
/// Dimension `8 * chunk + lane` accumulates into `sums[lane]` with a separate
/// multiply and add; changing that association changes score bits.
#[inline(always)]
fn accumulate_row_codes<const BITS: usize>(
    query: &[f32],
    codes: &[u8],
    levels: &[f32; 16],
) -> f32 {
    let mask = (1_u32 << BITS) - 1;
    let code_chunks = codes.chunks_exact(BITS);
    debug_assert!(code_chunks.remainder().is_empty());
    let mut sums = [0.0_f32; 8];
    let code_chunks =
        code_chunks.map(|chunk| -> [u8; BITS] { chunk.try_into().expect("exact code chunk") });
    for (query, bytes) in query.chunks_exact(8).zip(code_chunks) {
        let packed = bytes
            .iter()
            .rev()
            .fold(0_u32, |packed, byte| packed << 8 | u32::from(*byte));
        for (lane, (sum, value)) in sums.iter_mut().zip(query).enumerate() {
            let code = (packed >> (lane * BITS)) & mask;
            *sum += value * levels[code as usize];
        }
    }
    sums.into_iter().sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom};

    fn vectors(dimensions: usize, rows: usize) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|row| {
                (0..dimensions)
                    .map(|column| {
                        (((row + 1) * (column + 3)) as f32 * 0.017).sin()
                            + ((row + column + 7) as f32 * 0.013).cos()
                    })
                    .collect()
            })
            .collect()
    }

    fn open_verified(path: &Path) -> Result<MappedQuantizedVectors, QuantizedError> {
        // SAFETY: each test keeps its temporary file immutable while mapped.
        unsafe { MappedQuantizedVectors::open_mmap_verified(path) }
    }

    #[test]
    fn configuration_is_typed_and_fail_closed() {
        for bits in QuantizedConfig::MIN_BITS..=QuantizedConfig::MAX_BITS {
            assert_eq!(QuantizedConfig::with_bits(bits).unwrap().bits, bits);
        }
        assert!(matches!(
            QuantizedConfig::with_bits(1),
            Err(QuantizedError::InvalidBits(1))
        ));
        assert!(matches!(
            QuantizedConfig::with_bits(5),
            Err(QuantizedError::InvalidBits(5))
        ));
    }

    #[test]
    fn snapshots_are_deterministic_mapped_and_queryable() {
        for dimensions in [7, 10, 384] {
            let vectors = vectors(dimensions, 24);
            for bits in 2..=4 {
                let first = tempfile::NamedTempFile::new().unwrap();
                let second = tempfile::NamedTempFile::new().unwrap();
                let config = QuantizedConfig::with_bits(bits).unwrap();
                for path in [first.path(), second.path()] {
                    MappedQuantizedVectors::save(
                        path,
                        dimensions,
                        QuantizedMetric::Cosine,
                        config,
                        &vectors,
                    )
                    .unwrap();
                }
                assert_eq!(std::fs::read(first.path()).unwrap(), std::fs::read(second.path()).unwrap());
                let mapped = open_verified(first.path()).unwrap();
                assert!(mapped.is_checksum_verified());
                assert_eq!(mapped.len(), vectors.len());
                assert_eq!(mapped.metadata().dimensions, dimensions);
                assert!(mapped.metadata().padded_dimensions >= dimensions);
                let self_score = mapped.score(&vectors[7], 7).unwrap();
                assert!(self_score > 0.75, "{dimensions}d/{bits}-bit self score {self_score}");
            }
        }
    }

    #[test]
    fn dot_product_preserves_vector_magnitude() {
        let vectors = vec![vec![1.0; 16], vec![3.0; 16]];
        let file = tempfile::NamedTempFile::new().unwrap();
        MappedQuantizedVectors::save(
            file.path(),
            16,
            QuantizedMetric::DotProduct,
            QuantizedConfig::default(),
            &vectors,
        )
        .unwrap();
        let mapped = open_verified(file.path()).unwrap();
        assert!(mapped.score(&vectors[0], 1).unwrap() > mapped.score(&vectors[0], 0).unwrap());
    }

    #[test]
    fn four_bit_rows_use_close_to_one_eighth_the_f32_payload() {
        let vectors = vectors(384, 100);
        let file = tempfile::NamedTempFile::new().unwrap();
        MappedQuantizedVectors::save(
            file.path(),
            384,
            QuantizedMetric::Cosine,
            QuantizedConfig::default(),
            &vectors,
        )
        .unwrap();
        let mapped = open_verified(file.path()).unwrap();
        assert_eq!(mapped.packed_bytes(), 100 * (384 / 2 + 4));
        assert!(mapped.packed_bytes() * 7 < 100 * 384 * size_of::<f32>());
    }

    #[test]
    fn prepared_queries_are_bound_to_their_codec_identity() {
        let vectors = vectors(32, 8);
        let first = tempfile::NamedTempFile::new().unwrap();
        let second = tempfile::NamedTempFile::new().unwrap();
        MappedQuantizedVectors::save(
            first.path(),
            32,
            QuantizedMetric::Cosine,
            QuantizedConfig::new(4, 1).unwrap(),
            &vectors,
        )
        .unwrap();
        MappedQuantizedVectors::save(
            second.path(),
            32,
            QuantizedMetric::Cosine,
            QuantizedConfig::new(4, 2).unwrap(),
            &vectors,
        )
        .unwrap();
        let first = open_verified(first.path()).unwrap();
        let second = open_verified(second.path()).unwrap();
        let prepared = first.prepare_query(&vectors[0]).unwrap();
        assert!(matches!(
            second.score_prepared(&prepared, 0),
            Err(QuantizedError::IncompatibleQuery(_))
        ));
    }

    #[test]
    fn embedded_unaligned_region_roundtrips() {
        let vectors = vectors(24, 12);
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let prefix = [0xa5; 13];
        file.write_all(&prefix).unwrap();
        let image_bytes = MappedQuantizedVectors::write_to(
            &mut file,
            24,
            QuantizedMetric::Cosine,
            QuantizedConfig::default(),
            &vectors,
        )
        .unwrap();
        file.write_all(&[0x5a; 7]).unwrap();
        file.flush().unwrap();
        file.as_file().sync_all().unwrap();
        // SAFETY: the temporary container remains immutable while mapped.
        let mapped = unsafe {
            MappedQuantizedVectors::open_mmap_region(
                file.path(),
                prefix.len() as u64,
                image_bytes,
                true,
            )
        }
        .unwrap();
        assert_eq!(mapped.len(), vectors.len());
        assert!(mapped.score(&vectors[3], 3).unwrap() > 0.8);
    }

    #[test]
    fn four_bit_scan_preserves_deterministic_top_k_quality() {
        let vectors = vectors(64, 256);
        let file = tempfile::NamedTempFile::new().unwrap();
        MappedQuantizedVectors::save(
            file.path(),
            64,
            QuantizedMetric::Cosine,
            QuantizedConfig::default(),
            &vectors,
        )
        .unwrap();
        let mapped = open_verified(file.path()).unwrap();
        let query = vectors[137]
            .iter()
            .enumerate()
            .map(|(column, value)| value + ((column + 5) as f32 * 0.031).sin() * 0.03)
            .collect::<Vec<_>>();
        let query_norm = l2_norm(&query);
        let mut exact = vectors
            .iter()
            .enumerate()
            .map(|(row, vector)| {
                let score = query
                    .iter()
                    .zip(vector)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    / (query_norm * l2_norm(vector));
                (row, score)
            })
            .collect::<Vec<_>>();
        exact.sort_by(|left, right| right.1.total_cmp(&left.1));
        let prepared = mapped.prepare_query(&query).unwrap();
        let mut approximate = (0..mapped.len())
            .map(|row| (row, mapped.score_prepared(&prepared, row).unwrap()))
            .collect::<Vec<_>>();
        approximate.sort_by(|left, right| right.1.total_cmp(&left.1));
        let overlap = approximate[..10]
            .iter()
            .filter(|candidate| exact[..10].iter().any(|item| item.0 == candidate.0))
            .count();
        assert!(overlap >= 8, "recall@10 was {overlap}/10");
    }

    #[test]
    fn verified_open_rejects_corruption() {
        let vectors = vectors(32, 8);
        let file = tempfile::NamedTempFile::new().unwrap();
        MappedQuantizedVectors::save(
            file.path(),
            32,
            QuantizedMetric::Cosine,
            QuantizedConfig::default(),
            &vectors,
        )
        .unwrap();
        let mut writable = OpenOptions::new().write(true).open(file.path()).unwrap();
        writable.seek(SeekFrom::Start(HEADER_BYTES as u64 + 3)).unwrap();
        writable.write_all(&[0xff]).unwrap();
        writable.sync_all().unwrap();
        assert!(matches!(
            open_verified(file.path()),
            Err(QuantizedError::InvalidSnapshot(message)) if message.contains("checksum")
        ));
    }

    #[test]
    fn invalid_inputs_fail_before_query_or_persistence() {
        let file = tempfile::NamedTempFile::new().unwrap();
        assert!(matches!(
            MappedQuantizedVectors::save(
                file.path(),
                0,
                QuantizedMetric::Cosine,
                QuantizedConfig::default(),
                &[] as &[Vec<f32>],
            ),
            Err(QuantizedError::ZeroDimensions)
        ));
        assert!(matches!(
            MappedQuantizedVectors::save(
                file.path(),
                4,
                QuantizedMetric::Cosine,
                QuantizedConfig::default(),
                &[vec![1.0, 2.0]],
            ),
            Err(QuantizedError::DimensionMismatch { .. })
        ));
    }

    #[cfg(feature = "hnsw")]
    #[test]
    fn hnsw_traverses_quantized_rows_and_filters_before_top_k() {
        use fast_hnsw::distance::Cosine;
        use fast_hnsw::Builder;

        let vectors = vectors(32, 96);
        let mut index = Builder::new()
            .m(16)
            .ef_construction(100)
            .seed(42)
            .build(Cosine).unwrap();
        for vector in &vectors {
            index.insert(vector.clone()).unwrap();
        }
        let file = tempfile::NamedTempFile::new().unwrap();
        MappedQuantizedVectors::save(
            file.path(),
            32,
            QuantizedMetric::Cosine,
            QuantizedConfig::default(),
            &vectors,
        )
        .unwrap();
        let mapped = open_verified(file.path()).unwrap();
        let quantized = QuantizedHnsw::new(&index, &mapped).unwrap();
        assert_eq!(quantized.search(&vectors[17], 1, 96).unwrap()[0].id, 17);
        let filtered = quantized
            .search_filtered(&vectors[17], 5, 96, |id| id % 3 == 0)
            .unwrap();
        assert_eq!(filtered.len(), 5);
        assert!(filtered.iter().all(|result| result.id % 3 == 0));
    }

    #[cfg(feature = "hnsw")]
    #[test]
    fn hnsw_pairing_rejects_row_and_dimension_mismatches() {
        use fast_hnsw::distance::Cosine;
        use fast_hnsw::Builder;

        let vectors = vectors(16, 4);
        let file = tempfile::NamedTempFile::new().unwrap();
        MappedQuantizedVectors::save(
            file.path(),
            16,
            QuantizedMetric::Cosine,
            QuantizedConfig::default(),
            &vectors,
        )
        .unwrap();
        let mapped = open_verified(file.path()).unwrap();
        let mut wrong_rows = Builder::new().build(Cosine).unwrap();
        wrong_rows.insert(vec![1.0; 16]).unwrap();
        assert!(matches!(
            QuantizedHnsw::new(&wrong_rows, &mapped),
            Err(QuantizedError::IncompatibleIndex(_))
        ));

        let mut wrong_dimensions = Builder::new().build(Cosine).unwrap();
        for _ in 0..4 {
            wrong_dimensions.insert(vec![1.0; 8]).unwrap();
        }
        assert!(matches!(
            QuantizedHnsw::new(&wrong_dimensions, &mapped),
            Err(QuantizedError::IncompatibleIndex(_))
        ));
    }

    /// Byte-at-a-time scorer kept as the bit-exact reference for the per-row kernel.
    fn reference_row_score(
        mapped: &MappedQuantizedVectors,
        query: &PreparedQuantizedQuery,
        row: usize,
    ) -> f32 {
        let stride = mapped.metadata.row_stride;
        let codes = &mapped.image[mapped.codes_offset + row * stride..][..stride];
        let levels = &mapped.parameters.level_table[..1 << mapped.metadata.config.bits];
        let mut sums = [0.0_f32; 8];
        for (chunk, query) in query.rotated.chunks_exact(8).enumerate() {
            match mapped.metadata.config.bits {
                2 => {
                    for group in 0..2 {
                        let byte = codes[chunk * 2 + group];
                        for lane in 0..4 {
                            let code = usize::from((byte >> (lane * 2)) & 0x03);
                            sums[group * 4 + lane] += query[group * 4 + lane] * levels[code];
                        }
                    }
                }
                3 => {
                    let packed = u32::from(codes[chunk * 3])
                        | u32::from(codes[chunk * 3 + 1]) << 8
                        | u32::from(codes[chunk * 3 + 2]) << 16;
                    for lane in 0..8 {
                        let code = ((packed >> (lane * 3)) & 0x07) as usize;
                        sums[lane] += query[lane] * levels[code];
                    }
                }
                4 => {
                    for pair in 0..4 {
                        let code = codes[chunk * 4 + pair];
                        sums[pair * 2] += query[pair * 2] * levels[usize::from(code & 0x0f)];
                        sums[pair * 2 + 1] += query[pair * 2 + 1] * levels[usize::from(code >> 4)];
                    }
                }
                bits => unreachable!("unsupported test width {bits}"),
            }
        }
        sums.into_iter().sum::<f32>()
            * read_scale(&mapped.image, mapped.scales_offset, row).unwrap()
    }

    #[test]
    fn per_row_scorer_is_bit_identical_to_reference() {
        for dimensions in [7, 10, 384, 1_000] {
            let vectors = vectors(dimensions, 70);
            for metric in [QuantizedMetric::Cosine, QuantizedMetric::DotProduct] {
                for bits in QuantizedConfig::MIN_BITS..=QuantizedConfig::MAX_BITS {
                    let file = tempfile::NamedTempFile::new().unwrap();
                    MappedQuantizedVectors::save(
                        file.path(),
                        dimensions,
                        metric,
                        QuantizedConfig::with_bits(bits).unwrap(),
                        &vectors,
                    )
                    .unwrap();
                    let mapped = open_verified(file.path()).unwrap();
                    for query_row in [0, 17, 69] {
                        let prepared = mapped.prepare_query(&vectors[query_row]).unwrap();
                        for row in 0..mapped.len() {
                            assert_eq!(
                                mapped.score_prepared(&prepared, row).unwrap().to_bits(),
                                reference_row_score(&mapped, &prepared, row).to_bits(),
                                "{dimensions}d q{bits} {metric:?} query {query_row} row {row}"
                            );
                        }
                    }
                }
            }
        }
    }
}
