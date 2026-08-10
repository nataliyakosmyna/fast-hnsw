//! Single-embedding index with typed per-vector payload.
//!
//! A [`LabeledIndex`] wraps an [`Hnsw`] and stores one value of type `L`
//! alongside every inserted vector.  The payload can be:
//!
//! * a **class label** (`u32`, `i32`)
//! * a **text tag** (`String`)
//! * a **secondary embedding** (`Vec<f32>`) — for simple cross-modal lookup
//!   without a second index
//! * any other type that implements [`Payload`]
//!
//! For the case where the payload is itself a full searchable embedding in a
//! *different* space, use [`PairedIndex`](crate::paired::PairedIndex) instead,
//! which builds a second HNSW graph so you can search from either side.
//!
//! # Example — classification label
//!
//! ```rust
//! use fast_hnsw::labeled::LabeledIndex;
//! use fast_hnsw::distance::Euclidean;
//!
//! let mut idx: LabeledIndex<Euclidean, u32> = LabeledIndex::new(
//!     Default::default(), Euclidean,
//! )?;
//!
//! idx.insert(vec![1.0, 0.0], 0_u32).unwrap();   // class 0
//! idx.insert(vec![0.0, 1.0], 1_u32).unwrap();   // class 1
//! idx.insert(vec![0.8, 0.2], 0_u32).unwrap();   // class 0
//!
//! let results = idx.search(&[0.9, 0.1], 2, 20).unwrap();
//! for r in &results {
//!     println!("id={} dist={:.3} class={}", r.id, r.distance, r.payload);
//! }
//! # Ok::<(), fast_hnsw::Error>(())
//! ```
//!
//! # Example — text label + save/load
//!
//! ```rust,no_run
//! use fast_hnsw::labeled::LabeledIndex;
//! use fast_hnsw::distance::Euclidean;
//!
//! let mut idx: LabeledIndex<Euclidean, String> = LabeledIndex::new(
//!     Default::default(), Euclidean,
//! )?;
//! idx.insert(vec![1.0, 0.0], "cat".to_string()).unwrap();
//! idx.insert(vec![0.0, 1.0], "dog".to_string()).unwrap();
//!
//! idx.save("animals.hnsw").unwrap();
//!
//! let loaded = LabeledIndex::<Euclidean, String>::load("animals.hnsw", Euclidean).unwrap();
//! let hits = loaded.search(&[0.9, 0.1], 1, 10).unwrap();
//! println!("nearest: {}", hits[0].payload);  // "cat"
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::io;
use std::path::Path;
use std::cell::RefCell;

use crate::Builder;
use crate::distance::Distance;
use crate::error::Result;
use crate::hnsw::{Config, Hnsw, SearchWorkspace};
use crate::payload::Payload;
use crate::persist::{self, MappedPayloads};

// ─── Result type ─────────────────────────────────────────────────────────────

/// One result returned by [`LabeledIndex::search`].
pub struct LabeledResult<'a, L> {
    /// Zero-based insertion id.
    pub id:        usize,
    /// Distance from the query in the metric space of the index.
    pub distance:  f32,
    /// Reference to the payload associated with this vector.
    pub payload:   &'a L,
    /// The stored vector (borrowed from the index).
    pub embedding: &'a [f32],
}

/// One result from a [`MappedLabeledIndex`] search.
///
/// Fixed-width payloads are decoded from the mmap into the owned `payload`
/// field; vectors remain borrowed from the mapping.
pub struct MappedLabeledResult<'a, L> {
    pub id: usize,
    pub distance: f32,
    pub payload: L,
    pub embedding: &'a [f32],
}

// ─── LabeledIndex ─────────────────────────────────────────────────────────────

/// HNSW index with a typed payload attached to every vector.
///
/// ## Type parameters
/// * `D` — distance metric (e.g. [`Euclidean`](crate::distance::Euclidean))
/// * `L` — payload type (any type implementing [`Payload`])
pub struct LabeledIndex<D: Distance, L: Payload> {
    /// The underlying HNSW graph.  Exposed as `pub` so you can call
    /// `index.inner.stats()`, `index.inner.dim()`, etc. directly.
    pub inner:    Hnsw<D>,
    payloads:     Vec<L>,
}

impl<D: Distance, L: Payload> LabeledIndex<D, L> {
    // ─── Constructors ─────────────────────────────────────────────────────

    /// Create a new, empty labeled index.
    ///
    /// For ergonomic construction with method chaining, use
    /// [`Builder::build_labeled`].
    pub fn new(config: Config, metric: D) -> Result<Self> {
        let payloads = Vec::with_capacity(config.capacity);
        Ok(Self { inner: Hnsw::new(config, metric)?, payloads })
    }

    /// Build from an existing [`Builder`].
    ///
    /// ```rust
    /// use fast_hnsw::{Builder, labeled::LabeledIndex};
    /// use fast_hnsw::distance::Euclidean;
    ///
    /// let mut idx: LabeledIndex<Euclidean, String> = Builder::new()
    ///     .m(16)
    ///     .ef_construction(200)
    ///     .capacity(10_000)
    ///     .build_labeled(Euclidean).unwrap();
    /// ```
    pub fn from_builder(builder: Builder, metric: D) -> Result<Self> {
        let inner = builder.build(metric)?;
        let payloads = Vec::with_capacity(inner.config().capacity);
        Ok(Self { inner, payloads })
    }

    // ─── Mutation ─────────────────────────────────────────────────────────

    /// Insert a vector with its associated payload.
    ///
    /// Returns the zero-based id assigned to this vector (same as `self.len()
    /// - 1` after the call).  Both the vector and the payload are stored
    ///   internally and retrievable by id.
    ///
    /// # Panics
    /// Panics if `embedding.len()` differs from previously inserted vectors,
    /// or if the index was loaded with `load_mmap` (read-only).
    pub fn insert(&mut self, embedding: Vec<f32>, payload: L) -> Result<usize> {
        let id = self.inner.insert(embedding)?;
        debug_assert_eq!(id, self.payloads.len());
        self.payloads.push(payload);
        Ok(id)
    }

    // ─── Query ────────────────────────────────────────────────────────────

    /// Find the `k` approximate nearest neighbours of `query`.
    ///
    /// `ef` controls the recall/speed trade-off (`ef ≥ k`; larger → better
    /// recall).  Each result carries:
    /// * `id` — the vector's insertion index
    /// * `distance` — distance from `query` in the index's metric space
    /// * `payload` — borrowed reference to the stored payload
    /// * `embedding` — borrowed slice of the stored vector
    pub fn search<'a>(
        &'a self,
        query: &[f32],
        k:     usize,
        ef:    usize,
    ) -> Result<Vec<LabeledResult<'a, L>>> {
        self.inner
            .search(query, k, ef)?
            .into_iter()
            .map(|sr| Ok(LabeledResult {
                id:        sr.id,
                distance:  sr.distance,
                payload:   &self.payloads[sr.id],
                embedding: self.inner.get_vector(sr.id)?,
            }))
            .collect()
    }

    /// Search with filter-before-top-k eligibility over ids and payloads.
    pub fn search_filtered<'a, F>(
        &'a self,
        query: &[f32],
        k: usize,
        ef: usize,
        accepts: F,
    ) -> Result<Vec<LabeledResult<'a, L>>>
    where
        F: Fn(usize, &L) -> bool,
    {
        self.inner
            .search_filtered(query, k, ef, |id| accepts(id, &self.payloads[id]))?
            .into_iter()
            .map(|result| Ok(LabeledResult {
                id: result.id,
                distance: result.distance,
                payload: &self.payloads[result.id],
                embedding: self.inner.get_vector(result.id)?,
            }))
            .collect()
    }

    // ─── Deletion ─────────────────────────────────────────────────────────

    /// Soft-delete `id`.  See [`Hnsw::remove`] for the semantics.
    ///
    /// The payload is retained so ids stay stable; it becomes unreachable
    /// through search and is dropped by [`compacted`](Self::compacted).
    pub fn remove(&mut self, id: usize) -> bool {
        self.inner.remove(id)
    }

    /// Clear the tombstone on `id`.
    pub fn restore(&mut self, id: usize) -> bool {
        self.inner.restore(id)
    }

    /// Whether `id` has been tombstoned.
    pub fn is_deleted(&self, id: usize) -> bool {
        self.inner.is_deleted(id)
    }

    /// Number of tombstoned slots.
    pub fn deleted_count(&self) -> usize {
        self.inner.deleted_count()
    }

    /// Number of vectors still reachable by search.
    pub fn live_len(&self) -> usize {
        self.inner.live_len()
    }

    /// Rebuild without the deleted entries, carrying payloads across the
    /// renumbering.
    ///
    /// Required before [`save`](Self::save) once anything has been removed:
    /// the file format cannot record tombstones.
    pub fn compacted(&self, metric: D, seed: Option<u64>) -> Result<Self> {
        let (inner, old_ids) = self.inner.compacted(metric, seed)?;
        let payloads = old_ids
            .into_iter()
            .map(|old_id| self.payloads[old_id].clone())
            .collect();
        Ok(Self { inner, payloads })
    }

    // ─── Direct access ────────────────────────────────────────────────────

    /// Retrieve the payload for a specific id.
    ///
    /// # Panics
    /// Panics if `id >= self.len()`.
    pub fn get_payload(&self, id: usize) -> Result<&L> {
        self.payloads
            .get(id)
            .ok_or(crate::Error::IdOutOfBounds { id, len: self.payloads.len() })
    }



    /// Retrieve the stored embedding for a specific id.
    ///
    /// # Panics
    /// Panics if `id >= self.len()`.
    pub fn get_embedding(&self, id: usize) -> Result<&[f32]> { self.inner.get_vector(id) }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize { self.inner.len() }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }

    // ─── Persistence ──────────────────────────────────────────────────────

    /// Serialize the index and all payloads to a single binary file.
    ///
    /// The file can be reloaded with [`load`](Self::load) or
    /// [`load_mmap`](Self::load_mmap).
    ///
    /// ## File format
    /// The file uses the standard HNSW binary format (see [`persist`]) with a
    /// payload section appended after the graph.
    ///
    /// [`persist`]: crate::persist
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        persist::save_with_payload(&self.inner, &self.payloads, path)
    }

    /// Serialize a compact snapshot for read-only mmap serving.
    ///
    /// The compact format stores only neighbour ids in adjacency lists. Open
    /// it with [`load_mmap`](Self::load_mmap) or
    /// [`load_mmap_fixed`](Self::load_mmap_fixed), not [`load`](Self::load).
    pub fn save_compact(&self, path: impl AsRef<Path>) -> io::Result<()> {
        persist::save_compact_with_payload(&self.inner, &self.payloads, path)
    }

    /// Load a labeled index from a file, copying vector data into RAM.
    ///
    /// Use this for indexes that fit comfortably in memory.
    pub fn load(path: impl AsRef<Path>, metric: D) -> io::Result<Self> {
        let (inner, payloads) = persist::load_with_payload(path, metric)?;
        Ok(Self { inner, payloads })
    }

    /// Load a labeled index with the vector section **memory-mapped**.
    ///
    /// Vector data is not copied into RAM — the OS page cache handles
    /// residency.  Ideal for indexes larger than available RAM.
    ///
    /// ## Constraints
    /// * Inserting new vectors into a mmap-backed index will **panic**.
    /// * The file must remain on disk while the index is in use.
    pub fn load_mmap(path: impl AsRef<Path>, metric: D) -> io::Result<Self> {
        let (inner, payloads) = persist::load_mmap_with_payload(path, metric)?;
        Ok(Self { inner, payloads })
    }

    /// Load vectors, graph, and a non-zero fixed-width payload column from one
    /// read-only mapping.
    ///
    /// Payload values are decoded individually when accessed, avoiding the
    /// `Vec<L>` allocation performed by [`load_mmap`](Self::load_mmap).
    /// Variable-width payload types return [`io::ErrorKind::InvalidInput`].
    pub fn load_mmap_fixed(
        path: impl AsRef<Path>,
        metric: D,
    ) -> io::Result<MappedLabeledIndex<D, L>> {
        MappedLabeledIndex::load(path, metric)
    }
}

/// Read-only labeled index whose vectors, graph, and fixed-width payload
/// column share one file mapping.
pub struct MappedLabeledIndex<D: Distance, L: Payload> {
    pub inner: Hnsw<D>,
    payloads: MappedPayloads<L>,
}

impl<D: Distance, L: Payload> MappedLabeledIndex<D, L> {
    /// Open a fixed-width labeled index without materializing its payloads.
    pub fn load(path: impl AsRef<Path>, metric: D) -> io::Result<Self> {
        let (inner, payloads) = persist::load_mmap_with_fixed_payload(path, metric)?;
        Ok(Self { inner, payloads })
    }

    /// Search and decode only payloads attached to returned neighbors.
    pub fn search<'a>(
        &'a self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> io::Result<Vec<MappedLabeledResult<'a, L>>> {
        crate::hnsw::with_query_workspace(|workspace| {
            self.search_with_workspace(query, k, ef, workspace)
        })
    }

    /// Search while retaining traversal allocations in a caller-owned
    /// workspace for reuse by subsequent queries.
    pub fn search_with_workspace<'a>(
        &'a self,
        query: &[f32],
        k: usize,
        ef: usize,
        workspace: &mut SearchWorkspace,
    ) -> io::Result<Vec<MappedLabeledResult<'a, L>>> {
        self.search_map_with_workspace(query, k, ef, workspace, |result| result)
    }

    /// Search and transform decoded mapped results directly into the output type.
    ///
    /// This avoids materializing an intermediate `Vec<SearchResult>` or
    /// `Vec<MappedLabeledResult>` when an adapter immediately converts mapped
    /// results into its own public representation, and sizes the returned
    /// allocation for `T` directly.
    pub fn search_map_with_workspace<'a, T, M>(
        &'a self,
        query: &[f32],
        k: usize,
        ef: usize,
        workspace: &mut SearchWorkspace,
        mut map: M,
    ) -> io::Result<Vec<T>>
    where
        M: FnMut(MappedLabeledResult<'a, L>) -> T,
    {
        self.inner.try_search_with_workspace_map(
            query,
            k,
            ef,
            workspace,
            |result| {
                Ok(map(MappedLabeledResult {
                    id: result.id,
                    distance: result.distance,
                    payload: self.payloads.get(result.id)?,
                    embedding: self.inner.get_vector(result.id)?,
                }))
            },
        )
    }

    /// Search with filter-before-top-k eligibility over lazily decoded
    /// fixed-width payloads.
    pub fn search_filtered<'a, F>(
        &'a self,
        query: &[f32],
        k: usize,
        ef: usize,
        accepts: F,
    ) -> io::Result<Vec<MappedLabeledResult<'a, L>>>
    where
        F: Fn(usize, &L) -> bool,
    {
        // Reuse the calling thread's workspace rather than allocating a
        // visited-stamp array proportional to the index size per query.
        crate::hnsw::with_query_workspace(|workspace| {
            self.search_filtered_with_workspace(query, k, ef, &accepts, workspace)
        })
    }

    /// Filter-before-top-k search with reusable caller-owned traversal storage.
    pub fn search_filtered_with_workspace<'a, F>(
        &'a self,
        query: &[f32],
        k: usize,
        ef: usize,
        accepts: F,
        workspace: &mut SearchWorkspace,
    ) -> io::Result<Vec<MappedLabeledResult<'a, L>>>
    where
        F: Fn(usize, &L) -> bool,
    {
        self.search_filtered_map_with_workspace(
            query,
            k,
            ef,
            accepts,
            workspace,
            |result| result,
        )
    }

    /// Filter-before-top-k search that transforms decoded results directly.
    pub fn search_filtered_map_with_workspace<'a, T, F, M>(
        &'a self,
        query: &[f32],
        k: usize,
        ef: usize,
        accepts: F,
        workspace: &mut SearchWorkspace,
        mut map: M,
    ) -> io::Result<Vec<T>>
    where
        F: Fn(usize, &L) -> bool,
        M: FnMut(MappedLabeledResult<'a, L>) -> T,
    {
        let decode_error = RefCell::new(None);
        let results = self.inner.try_search_filtered_with_workspace_map(
            query,
            k,
            ef,
            |id| match self.payloads.get(id) {
                Ok(payload) => accepts(id, &payload),
                Err(error) => {
                    *decode_error.borrow_mut() = Some(error);
                    false
                }
            },
            workspace,
            |result| {
                Ok(map(MappedLabeledResult {
                    id: result.id,
                    distance: result.distance,
                    payload: self.payloads.get(result.id)?,
                    embedding: self.inner.get_vector(result.id)?,
                }))
            },
        );
        if let Some(error) = decode_error.into_inner() {
            return Err(error);
        }

        results
    }

    /// Decode one payload directly from the mapped column.
    pub fn get_payload(&self, id: usize) -> io::Result<L> {
        self.payloads.get(id)
    }

    pub fn get_embedding(&self, id: usize) -> io::Result<&[f32]> {
        Ok(self.inner.get_vector(id)?)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ─── Builder extension ────────────────────────────────────────────────────────

impl Builder {
    /// Consume the builder and create an empty [`LabeledIndex`].
    ///
    /// ```rust
    /// use fast_hnsw::{Builder, labeled::LabeledIndex};
    /// use fast_hnsw::distance::Euclidean;
    ///
    /// let mut idx: LabeledIndex<Euclidean, u32> = Builder::new()
    ///     .m(16)
    ///     .ef_construction(200)
    ///     .build_labeled(Euclidean).unwrap();
    /// idx.insert(vec![1.0, 2.0], 42_u32).unwrap();
    /// ```
    pub fn build_labeled<D: Distance, L: Payload>(
        self,
        metric: D,
    ) -> Result<LabeledIndex<D, L>> {
        LabeledIndex::from_builder(self, metric)
    }
}
