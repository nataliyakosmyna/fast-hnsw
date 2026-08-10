//! Core HNSW index — optimised implementation.
//!
//! ## Optimisations (in order of measured impact)
//!
//! ### 1 · Simple sort-and-truncate for reverse-update pruning  *(biggest win)*
//!
//! When we add a bidirectional edge `q ↔ nb` and `nb`'s connection list
//! exceeds `M`, we must shrink it back to `M`.  The naïve approach recomputes
//! all `M+1` distances from scratch (cache-miss-dominated) and then runs the
//! O(M²/2) heuristic.  Instead:
//!
//! * **Connection lists store `(u32, f32)` pairs** — the neighbour id and the
//!   distance from *this node* to that neighbour.  The distance is already
//!   known at edge-addition time (symmetric metric), so storing it is free.
//! * **Reverse-update prune** = `sort_unstable_by(dist) + truncate(M)` — zero
//!   new distance computations, ~25 ns vs ~2 000 ns in-cache (measured),
//!   ~11 µs with real L3-miss rates × 16 prunes/insert ≈ **170 µs/insert saved**
//!   at n=10k/dim=128.
//!
//! ### 2 · Iterate `scratch.out` directly in heuristic selection
//!
//! `scratch.out` is already sorted closest-first after `search_layer`.
//! Previously `select_neighbours_heuristic` rebuilt a `BinaryHeap` from it
//! (O(ef log ef) + 1 allocation).  Now we iterate the slice directly: 321 ns
//! → 122 ns measured, plus the allocation is gone.
//!
//! ### 3 · Early return in heuristic when |candidates| ≤ M
//!
//! When fewer candidates exist than there are slots, **all candidates are
//! automatically diverse** — no pairwise distance check is needed.  Copy
//! directly to `select_buf` in O(M) time (50 ns vs 10 000 ns).  Triggers at
//! every upper-layer descent (ef=1) and during early index build.
//!
//! ### 4 · `VecStore` — flat contiguous vector storage
//!
//! All feature vectors in one `Vec<f32>`, stride = `dim`.  One pointer
//! dereference instead of two per distance call.
//!
//! ### 5 · Lane-folded distance kernels
//!
//! Every built-in metric folds into eight independent accumulators.  Float
//! addition is not associative, so a plain `zip().sum()` is one serial
//! dependency chain as long as the vector dimension and the compiler may not
//! re-order it; splitting the fold lets the vectorizer emit SIMD adds without
//! a platform-specific SIMD dependency.  Measured 1.9× at dim 128 and 4.1× at
//! dim 768 versus the serial fold.
//!
//! ### 6 · `VisitedTracker` — O(1) generation-counter visited set
//!
//! Replaces `HashSet::with_capacity(ef*4)` (1 703 ns/call) with a stamp array
//! (105 ns/call) — 16× faster, zero allocation after construction.
//!
//! ### 7 · `Scratch` / `SearchWorkspace` — reusable query storage
//!
//! Both `BinaryHeap`s in `search_layer` are cleared (not reallocated) between
//! calls. `SearchWorkspace` lets repeated `&self` queries retain their visited
//! stamps, heaps, and entry buffer. `ep_buf` is swapped with `scratch.out` via
//! `std::mem::swap` — zero copy between layers.
//!
//! ### 8 · `u32` + pre-allocated connection `Vec`s
//!
//! `u32` IDs halve connection-list memory.  Each inner `Vec` is pre-created
//! with `Vec::with_capacity(m_max)`.
//!
//! ### 9 · Triangle-inequality shortcut in heuristic selection
//!
//! If `d(q,s) > 2·d(q,e)`, then `d(e,s) > d(q,e)` by the triangle
//! inequality — the heuristic condition is trivially satisfied without
//! computing the actual distance.  Measured 36% reduction in pairwise
//! distance computations.
//!
//! ### 10 · `select_buf` / `pruned_buf` reuse
//!
//! Pre-allocated `Vec<(usize, f32)>` fields in `Hnsw` hold the results of
//! `select_neighbours_*`; callers read from `self.select_buf` directly,
//! eliminating the one remaining per-call `Vec` allocation.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::distance::Distance;
use crate::error::{Error, Result};
use crate::heap::DistId;

// ─── Memory-mapped vector backing ────────────────────────────────────────────

/// A live memory-mapping of the vector section of an index file.
///
/// Kept alive inside [`VecStore`] so the mmap is not unmapped while the index
/// is in use.  The raw pointer is derived from the `Mmap` bytes; both the
/// pointer and the `Mmap` live inside the same `Arc` so they are guaranteed
/// to share the same lifetime.
///
/// # Safety invariants
/// * `ptr` always points into `mmap`'s mapped region.
/// * `len` is the total number of `f32` values in that region
///   (`n_vectors × dim`).
/// * The region is read-only — no writes ever go through this pointer.
pub(crate) struct MmapBacking {
    /// Keeps the file mapping alive.  Dropped when the last `Arc` clone is
    /// released (i.e. when the `Hnsw` is dropped).
    pub(crate) _mmap: Arc<memmap2::Mmap>,
    /// Pointer to the first `f32` in the mapped vector section.
    pub(crate) ptr: *const f32,
    /// Total number of `f32` values available through `ptr`.
    pub(crate) len: usize,
}

// The pointer never leaves the struct boundary through a mutable reference,
// and the mapping is read-only, so both Send and Sync are safe.
unsafe impl Send for MmapBacking {}
unsafe impl Sync for MmapBacking {}

// ─── Flat vector store ────────────────────────────────────────────────────────

/// All feature vectors packed contiguously: vector `i` occupies
/// `data[i*dim .. (i+1)*dim]` (owned mode) or the corresponding slice of the
/// mmap'd region (mmap mode).
///
/// ## Owned mode
/// Created by normal [`Hnsw::insert`] calls.  Vectors live in a `Vec<f32>`
/// that grows as new items are added.
///
/// ## Mmap mode
/// Created by [`Hnsw::load_mmap`].  Vectors are read directly from the
/// memory-mapped file region — no heap copy.  The file must remain on disk
/// and the [`MmapBacking`] must stay alive (it is stored in `mmap`).
/// Inserts into a mmap-backed index are not allowed.
pub(crate) struct VecStore {
    /// Owned vector data (empty in mmap and deferred modes).
    pub(crate) data: Vec<f32>,
    /// Optional memory-mapped backing (non-`None` in mmap mode).
    pub(crate) mmap: Option<MmapBacking>,
    /// Vector count when the vectors are not resident at all — they live on
    /// disk and are fetched per access by
    /// [`PreadIndex`](crate::pread::PreadIndex). The store then knows how many
    /// vectors exist and their width, but holds none of their bytes.
    pub(crate) deferred: Option<usize>,
    pub(crate) dim:  usize,
}

impl VecStore {
    pub(crate) fn new(dim: usize, capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity.saturating_mul(dim.max(1))),
            mmap: None,
            deferred: None,
            dim,
        }
    }

    /// A store that knows its shape but holds no vector bytes.
    pub(crate) fn deferred(dim: usize, count: usize) -> Self {
        Self {
            data: Vec::new(),
            mmap: None,
            deferred: Some(count),
            dim,
        }
    }

    /// Build a mmap-backed store from a live mapping and the byte offset where
    /// the vector data begins.
    pub(crate) fn from_mmap(
        mmap: Arc<memmap2::Mmap>,
        vec_offset: usize,
        len: usize,
        dim: usize,
    ) -> Self {
        // SAFETY: vec_offset + len*4 is guaranteed to be within the mapped
        // region by the caller (checked in `load_mmap`).
        let ptr = unsafe { mmap.as_ptr().add(vec_offset) as *const f32 };
        Self {
            data: Vec::new(),
            mmap: Some(MmapBacking { _mmap: mmap, ptr, len }),
            deferred: None,
            dim,
        }
    }

    /// # Invariants
    /// The caller must already have rejected read-only stores; `Hnsw::insert`
    /// returns [`Error::ReadOnly`] before reaching here, so this is a
    /// debug-only guard against an internal mistake, not input validation.
    pub(crate) fn push(&mut self, v: Vec<f32>) {
        debug_assert!(self.mmap.is_none(), "push into a read-only store");
        debug_assert_eq!(v.len(), self.dim);
        self.data.extend_from_slice(&v);
    }

    /// # Invariants
    /// `id` must be in range and the store resident. Both are checked by the
    /// public `Hnsw::get_vector`, which returns an error instead; internal
    /// callers pass ids that traversal already validated. The bounds check on
    /// the mapped branch below is kept regardless, because it guards `unsafe`
    /// pointer arithmetic — dropping it would turn a bug into undefined
    /// behaviour rather than a crash.
    #[inline(always)]
    pub(crate) fn get(&self, id: usize) -> &[f32] {
        debug_assert!(self.deferred.is_none(), "vectors are not resident");
        let s = id * self.dim;
        match &self.mmap {
            None => &self.data[s..s + self.dim],
            Some(mb) => {
                // The owned branch above is bounds-checked by slicing, and
                // `get` is reachable from the safe public `Hnsw::get_vector`,
                // so the mapped branch must be checked too — an unchecked
                // pointer offset here would turn an out-of-range id into
                // undefined behaviour rather than a panic.
                assert!(
                    s <= mb.len && self.dim <= mb.len - s,
                    "vector id {id} is out of bounds for a memory-mapped index \
                     holding {} vectors",
                    mb.len.checked_div(self.dim).unwrap_or(0),
                );
                // SAFETY: the assertion above establishes that
                // `s + self.dim <= mb.len`, and `MmapBacking`'s invariants
                // guarantee `ptr` addresses `len` readable `f32` values for
                // as long as the backing `Arc<Mmap>` is alive.
                unsafe { std::slice::from_raw_parts(mb.ptr.add(s), self.dim) }
            }
        }
    }

    /// `Some(count)` when the vectors live on disk rather than in memory.
    #[inline]
    pub(crate) fn deferred_len(&self) -> Option<usize> {
        self.deferred
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        if let Some(count) = self.deferred {
            return count;
        }
        if self.dim == 0 {
            return 0;
        }
        match &self.mmap {
            None     => self.data.len() / self.dim,
            Some(mb) => mb.len / self.dim,
        }
    }

    /// The whole vector section as one contiguous `f32` slice.
    ///
    /// `None` when the vectors are not resident (deferred/`pread` mode), where
    /// there is nothing contiguous to hand out.
    ///
    /// Only the BLAS `sgemv` path needs a contiguous view; every other caller
    /// goes through `get`.
    #[cfg(feature = "blas")]
    #[inline]
    pub(crate) fn as_slice(&self) -> Option<&[f32]> {
        if self.deferred.is_some() {
            return None;
        }
        match &self.mmap {
            None => Some(&self.data),
            // SAFETY: `MmapBacking`'s invariants guarantee `ptr` addresses
            // `len` readable `f32` values for as long as the mapping is alive,
            // which it is for the lifetime of this borrow.
            Some(mb) => Some(unsafe { std::slice::from_raw_parts(mb.ptr, mb.len) }),
        }
    }

    /// Returns the whole vector section as a flat byte slice (for writing).
    /// # Invariants
    /// Only reached from the persistence writers, which are unreachable for a
    /// non-resident index (`PreadIndex` exposes no `save`).
    pub(crate) fn as_bytes(&self) -> &[u8] {
        debug_assert!(self.deferred.is_none(), "serialising a non-resident store");
        let (ptr, len) = match &self.mmap {
            None => (self.data.as_ptr(), self.data.len()),
            Some(mb) => (mb.ptr, mb.len),
        };
        // SAFETY: f32 has no padding; any bit pattern is valid; alignment is
        // satisfied because both owned and mapped vector sections are
        // represented as aligned f32 storage.
        unsafe {
            std::slice::from_raw_parts(
                ptr as *const u8,
                len * std::mem::size_of::<f32>(),
            )
        }
    }
}

// ─── Graph storage ───────────────────────────────────────────────────────────

pub(crate) type Edge = (u32, f32);
pub(crate) type OwnedGraph = Vec<Vec<Vec<Edge>>>;

/// Storage boundary for HNSW adjacency lists.
///
/// Today the graph is owned and mutable. Search deliberately accesses it
/// through [`GraphStore::neighbours`] so a read-only memory-mapped
/// implementation can decode on-disk edges without changing the search
/// algorithm or relying on the alignment/layout of `(u32, f32)`.
pub(crate) enum GraphStore {
    Owned(OwnedGraph),
    Mapped(MappedGraph),
}

pub(crate) struct MappedGraph {
    mmap: Arc<memmap2::Mmap>,
    levels_offset: usize,
    offsets_offset: usize,
    node_count: usize,
    edge_stride: usize,
}

pub(crate) enum Neighbours<'a> {
    Owned(std::iter::Copied<std::slice::Iter<'a, Edge>>),
    Mapped(MappedNeighbours<'a>),
}

pub(crate) struct MappedNeighbours<'a> {
    bytes: &'a [u8],
    position: usize,
    edge_stride: usize,
}

impl Iterator for Neighbours<'_> {
    type Item = Edge;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Owned(inner) => inner.next(),
            Self::Mapped(inner) => inner.next(),
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Owned(inner) => inner.size_hint(),
            Self::Mapped(inner) => inner.size_hint(),
        }
    }
}

impl ExactSizeIterator for Neighbours<'_> {}

impl Iterator for MappedNeighbours<'_> {
    type Item = Edge;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let edge = self
            .bytes
            .get(self.position..self.position + self.edge_stride)?;
        self.position += self.edge_stride;
        let id = u32::from_le_bytes(edge[..4].try_into().unwrap());
        // Compact snapshots omit build-time edge distances. Search computes
        // query-to-node distances from vectors and never consumes this value.
        let distance = if self.edge_stride == 8 {
            f32::from_le_bytes(edge[4..8].try_into().unwrap())
        } else {
            0.0
        };
        Some((id, distance))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.bytes.len() - self.position) / self.edge_stride;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for MappedNeighbours<'_> {}

impl GraphStore {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::Owned(Vec::with_capacity(capacity))
    }

    pub(crate) fn from_owned(owned: OwnedGraph) -> Self {
        Self::Owned(owned)
    }

    pub(crate) fn from_mmap(
        mmap: Arc<memmap2::Mmap>,
        levels_offset: usize,
        node_count: usize,
        edge_stride: usize,
    ) -> Self {
        Self::Mapped(MappedGraph {
            mmap,
            levels_offset,
            offsets_offset: levels_offset + node_count * 4,
            node_count,
            edge_stride,
        })
    }

    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        match self {
            Self::Owned(graph) => graph.len(),
            Self::Mapped(graph) => graph.node_count,
        }
    }

    #[inline]
    pub(crate) fn level_count(&self, node: usize) -> usize {
        match self {
            Self::Owned(graph) => graph[node].len(),
            Self::Mapped(graph) => graph.level(node) + 1,
        }
    }

    #[inline]
    pub(crate) fn neighbour_count(&self, node: usize, layer: usize) -> usize {
        match self.neighbours(node, layer) {
            Some(neighbours) => neighbours.len(),
            None => 0,
        }
    }

    #[inline]
    pub(crate) fn neighbours(&self, node: usize, layer: usize) -> Option<Neighbours<'_>> {
        match self {
            Self::Owned(graph) => graph
                .get(node)
                .and_then(|layers| layers.get(layer))
                .map(|edges| Neighbours::Owned(edges.iter().copied())),
            Self::Mapped(graph) => graph.neighbours(node, layer).map(Neighbours::Mapped),
        }
    }

    #[inline]
    fn push_node(&mut self, connections: Vec<Vec<Edge>>) {
        match self {
            Self::Owned(graph) => graph.push(connections),
            // Unreachable: `Hnsw::insert` returns `Error::ReadOnly` before
            // reaching the graph.
            Self::Mapped(_) => unreachable!("push into a read-only graph"),
        }
    }

    #[inline]
    fn neighbours_mut(&mut self, node: usize, layer: usize) -> &mut Vec<Edge> {
        match self {
            Self::Owned(graph) => &mut graph[node][layer],
            // Unreachable for the same reason as `push_node`.
            Self::Mapped(_) => unreachable!("mutating a read-only graph"),
        }
    }
}

impl MappedGraph {
    #[inline(always)]
    fn level(&self, node: usize) -> usize {
        let start = self.levels_offset + node * 4;
        u32::from_le_bytes(self.mmap[start..start + 4].try_into().unwrap()) as usize
    }

    #[inline(always)]
    fn node_offset(&self, node: usize) -> usize {
        let start = self.offsets_offset + node * 8;
        u64::from_le_bytes(self.mmap[start..start + 8].try_into().unwrap()) as usize
    }

    #[inline]
    fn neighbours(&self, node: usize, layer: usize) -> Option<MappedNeighbours<'_>> {
        if node >= self.node_count || layer > self.level(node) {
            return None;
        }

        let mut position = self.node_offset(node);
        for _ in 0..layer {
            let count =
                u32::from_le_bytes(self.mmap[position..position + 4].try_into().unwrap()) as usize;
            position += 4 + count * self.edge_stride;
        }

        let count =
            u32::from_le_bytes(self.mmap[position..position + 4].try_into().unwrap()) as usize;
        let start = position + 4;
        let end = start + count * self.edge_stride;
        Some(MappedNeighbours {
            bytes: &self.mmap[start..end],
            position: 0,
            edge_stride: self.edge_stride,
        })
    }
}

// ─── Generation-counter visited-set ──────────────────────────────────────────

/// O(1) visited-node tracker.  Each search query increments `current`; a node
/// `i` is "visited" iff `stamps[i] == current`.  `begin()` is a single
/// integer increment — no allocation after construction.
pub(crate) struct VisitedTracker {
    stamps:  Vec<u32>,
    current: u32,
}

impl VisitedTracker {
    pub(crate) fn new(capacity: usize) -> Self {
        Self { stamps: vec![0u32; capacity], current: 1 }
    }

    #[inline]
    pub(crate) fn begin(&mut self) {
        self.current = self.current.wrapping_add(1);
        if self.current == 0 {
            self.stamps.fill(0);
            self.current = 1;
        }
    }

    #[inline]
    fn reserve_nodes(&mut self, capacity: usize) {
        if self.stamps.len() < capacity {
            self.stamps.resize(capacity, 0);
        }
    }

    /// Returns `true` if `id` was **not** previously visited, and marks it.
    #[inline]
    pub(crate) fn visit(&mut self, id: usize) -> bool {
        if id >= self.stamps.len() {
            self.stamps.resize(id * 2 + 1, 0);
        }
        if self.stamps[id] == self.current {
            false
        } else {
            self.stamps[id] = self.current;
            true
        }
    }
}

// ─── Reusable search scratch space ───────────────────────────────────────────

/// Pre-allocated candidate min-heap + result max-heap + sorted output buffer.
/// Stored inside `Hnsw` and *cleared* (not reallocated) between `search_layer`
/// calls during `insert`. Repeated immutable searches may retain one through
/// [`SearchWorkspace`].
struct Scratch {
    candidates:  BinaryHeap<Reverse<DistId>>,
    results:     BinaryHeap<DistId>,
    results_cap: usize,
    /// Sorted-closest-first output written by `finish()`.
    pub out: Vec<DistId>,
}

impl Scratch {
    fn new(ef: usize) -> Self {
        Self {
            candidates:  BinaryHeap::with_capacity(ef * 2 + 1),
            results:     BinaryHeap::with_capacity(ef + 1),
            results_cap: ef,
            out:         Vec::with_capacity(ef),
        }
    }

    #[inline]
    fn begin(&mut self, ef: usize) {
        self.candidates.clear();
        self.results.clear();
        self.results_cap = ef;
    }

    fn reserve_ef(&mut self, ef: usize) {
        let candidate_capacity = ef.saturating_mul(2).saturating_add(1);
        if self.candidates.capacity() < candidate_capacity {
            self.candidates
                .reserve(candidate_capacity.saturating_sub(self.candidates.len()));
        }
        if self.results.capacity() < ef.saturating_add(1) {
            self.results
                .reserve(ef.saturating_add(1).saturating_sub(self.results.len()));
        }
        if self.out.capacity() < ef {
            self.out.reserve(ef.saturating_sub(self.out.len()));
        }
    }

    /// Admit `d` both as a traversal candidate and as a result.
    ///
    /// Used for entry points and for every neighbour accepted by an unfiltered
    /// layer search — in that mode the two roles always coincide.  The filtered
    /// search separates them via [`Scratch::push_navigation`] and
    /// [`Scratch::push_result`].
    #[inline]
    fn push_candidate(&mut self, d: DistId) {
        self.candidates.push(Reverse(d));
        self.results.push(d);
        if self.results.len() > self.results_cap { self.results.pop(); }
    }

    #[inline]
    fn push_navigation(&mut self, d: DistId) {
        self.candidates.push(Reverse(d));
    }

    #[inline]
    fn push_result(&mut self, d: DistId) {
        self.results.push(d);
        if self.results.len() > self.results_cap {
            self.results.pop();
        }
    }

    #[inline]
    fn pop_candidate(&mut self) -> Option<DistId> {
        self.candidates.pop().map(|Reverse(x)| x)
    }

    #[inline]
    fn worst_result_dist(&self) -> Option<f32> {
        self.results.peek().map(|x| x.dist)
    }

    #[inline]
    fn results_len(&self) -> usize { self.results.len() }

    fn finish(&mut self) {
        self.out.clear();
        while let Some(d) = self.results.pop() { self.out.push(d); }
        self.out.reverse(); // max-heap → farthest-first; reverse → closest-first
    }
}

/// Reusable, caller-owned storage for allocation-free repeated searches.
///
/// A workspace is mutable and therefore belongs to one query thread at a
/// time. The index itself remains shared through `&Hnsw`; give each concurrent
/// worker its own workspace. It grows automatically if either the index or
/// `ef` exceeds the initial capacities, then retains those allocations.
pub struct SearchWorkspace {
    visited: VisitedTracker,
    scratch: Scratch,
    entry_points: Vec<DistId>,
}

impl SearchWorkspace {
    /// Pre-allocate storage for an expected node count and search `ef`.
    pub fn new(node_capacity: usize, ef_capacity: usize) -> Self {
        Self {
            visited: VisitedTracker::new(node_capacity),
            scratch: Scratch::new(ef_capacity),
            entry_points: Vec::with_capacity(ef_capacity),
        }
    }

    fn prepare(&mut self, node_capacity: usize, ef_capacity: usize) {
        self.visited.reserve_nodes(node_capacity);
        self.scratch.reserve_ef(ef_capacity);
        self.entry_points.clear();
        if self.entry_points.capacity() < ef_capacity {
            self.entry_points.reserve(ef_capacity);
        }
    }
}

impl Default for SearchWorkspace {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

thread_local! {
    /// Scratch reused by the convenience search entry points.
    ///
    /// Without this, every `search()` call allocates and zeroes a visited-stamp
    /// array proportional to the index size; at a million vectors that setup
    /// cost dominated the traversal itself.  The workspace is per-thread, so
    /// concurrent queries still never share mutable state.
    static QUERY_WORKSPACE: std::cell::RefCell<SearchWorkspace> =
        std::cell::RefCell::new(SearchWorkspace::default());
}

/// Run `f` with the calling thread's shared query workspace.
///
/// Falls back to a private workspace when the thread-local is already borrowed,
/// which happens if a user-supplied distance or filter callback issues another
/// search on the same thread.  Re-entrancy therefore costs an allocation rather
/// than panicking.
pub(crate) fn with_query_workspace<R>(f: impl FnOnce(&mut SearchWorkspace) -> R) -> R {
    QUERY_WORKSPACE.with(|cell| match cell.try_borrow_mut() {
        Ok(mut workspace) => f(&mut workspace),
        Err(_) => f(&mut SearchWorkspace::default()),
    })
}

// ─── Prune strategy ───────────────────────────────────────────────────────────

/// Controls how an existing node's connection list is shrunk back to `M`
/// entries when it overflows after a bidirectional edge `q ↔ nb` is added
/// during [`Hnsw::insert`].
///
/// # Background
///
/// Every insert adds M bidirectional edges.  For each selected neighbour `nb`,
/// the reverse edge `nb → q` is appended to `nb`'s connection list.  If `nb`
/// already had `M` connections its list grows to `M + 1` and must be pruned
/// back to `M`.  The two strategies differ in *which* edge is dropped:
///
/// | | [`Simple`](PruneStrategy::Simple) | [`Heuristic`](PruneStrategy::Heuristic) |
/// |---|---|---|
/// | Work per prune | Sort `M+1` stored `f32`s + truncate | O(M²/2) pairwise distance computations |
/// | New distance calls | **0** | up to M²/2, ~40% skipped by triangle shortcut |
/// | Speed (M=16, dim=128) | ~25 ns | ~1–11 µs depending on cache state |
/// | Recall impact | −0 to −1 pp vs Heuristic | full quality |
///
/// # Which to choose
///
/// **`Simple` (default)** — use when insert throughput is the priority.
/// Equivalent to what most production HNSW libraries (faiss, hnsw_rs) do for
/// reverse-update pruning.  The small recall gap can be recovered by raising
/// `ef` at query time.
///
/// **`Heuristic`** — use when recall quality is non-negotiable or you need
/// a like-for-like algorithmic comparison.  Runs the full paper Algorithm 4
/// for *every* edge in the graph (not just the new node's own connections),
/// at the cost of slower inserts on high-dimensional data.
///
/// # Why `Heuristic` is not as expensive as it looks
///
/// Connection lists store `(neighbour_id: u32, dist_from_this_node: f32)`.
/// Because the distance is recorded at edge-add time (symmetric metric, free),
/// the M distance *recomputations* that a naïve heuristic prune would require
/// are completely eliminated.  Only the inter-neighbour pairwise checks remain.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PruneStrategy {
    /// **Sort-and-truncate** (default, fastest).
    ///
    /// Sorts the `M+1`-entry connection list by the stored per-edge distance
    /// and truncates to `M`.  The farthest neighbour by raw distance is
    /// dropped.
    ///
    /// - **Zero new distance computations** — distances are already stored as
    ///   `f32` values alongside the neighbour id.
    /// - Benchmark: ~25 ns per prune call (in-cache), effectively 0 at L3
    ///   miss rates because no vector data is touched.
    /// - Recall cost: ≈ 0–1 pp on high-dimensional data (dim ≥ 64) where a
    ///   cluster of nearby nodes can crowd out a more structurally useful but
    ///   slightly farther connection.
    #[default]
    Simple,

    /// **Full Algorithm 4 (heuristic)** using stored distances.
    ///
    /// Runs the paper's diversity-based selection for every reverse-update
    /// prune: candidate `e` is kept only if `d(node, e) ≤ d(e, s)` for every
    /// already-selected neighbour `s` — i.e. the node is closer to `e` than
    /// any currently-selected neighbour is.  This ensures the final `M`
    /// connections are spread across the neighbourhood rather than all
    /// pointing in the same direction.
    ///
    /// **Stored-distance optimisation**: `d(node, e)` for each candidate is
    /// read directly from the stored `f32` in the connection list — no
    /// distance recomputation.  Only the pairwise `d(e, s)` checks are
    /// computed fresh, and ~60% of those are eliminated by the
    /// triangle-inequality shortcut (`d(node,s) > 2·d(node,e)` guarantees
    /// the accept condition without touching vector data).
    ///
    /// - Benchmark: ~1 µs per prune in L2 cache, ~11 µs at L3-miss rates
    ///   (M = 16, dim = 128).  × 16 prunes/insert ≈ 170 µs extra per insert
    ///   on large, high-dimensional indexes.
    /// - Recall: full quality; recovers the gap vs [`Simple`](PruneStrategy::Simple).
    Heuristic,
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Build-time parameters for an [`Hnsw`] index.
///
/// Construct via [`Builder`](crate::Builder) for a more ergonomic API.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct Config {
    /// Max bidirectional links per non-zero layer (must be ≥ 2).
    pub m: usize,
    /// Override for layer-0 link limit (`None` → `2 * M`).
    pub m0: Option<usize>,
    /// Beam width during construction.  Higher → better graph quality,
    /// slower inserts.
    pub ef_construction: usize,
    /// Use heuristic neighbour selection (Algorithm 4) for the *new node's
    /// own* M connections.  Recommended; improves recall.  Setting this to
    /// `false` uses the simpler greedy strategy (Algorithm 3).
    pub use_heuristic: bool,
    /// `extendCandidates` flag from §4 Algorithm 4 of the paper.
    /// Adds the neighbours-of-candidates to the candidate set before
    /// heuristic selection.  Usually not needed; disabled by default.
    pub extend_candidates: bool,
    /// `keepPrunedConnections` flag from §4 Algorithm 4.
    /// When the heuristic rejects a candidate, pad the result with rejected
    /// candidates (nearest-first) until M is reached.  Improves recall on
    /// sparse graphs; enabled by default.
    pub keep_pruned: bool,
    /// Strategy for pruning an *existing* node's connection list when it
    /// overflows after a bidirectional edge is added.
    ///
    /// See [`PruneStrategy`] for a full comparison.  Defaults to
    /// [`PruneStrategy::Simple`] (fastest; zero new distance computations).
    pub prune_strategy: PruneStrategy,
    /// Expected number of vectors — pre-allocation hint only.  Has no effect
    /// on correctness.
    pub capacity: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            m: 16,
            m0: None,
            ef_construction: 200,
            use_heuristic: true,
            extend_candidates: false,
            keep_pruned: true,
            prune_strategy: PruneStrategy::Simple,
            capacity: 0,
        }
    }
}

impl Config {
    /// Reject parameters that cannot produce a usable graph.
    ///
    /// Called by every constructor, so an invalid `Config` is reported rather
    /// than asserted.
    pub fn validate(&self) -> Result<()> {
        if self.m < 2 {
            return Err(Error::InvalidConfig {
                parameter: "m",
                reason: format!("must be at least 2, got {}", self.m),
            });
        }
        if self.ef_construction < self.m {
            return Err(Error::InvalidConfig {
                parameter: "ef_construction",
                reason: format!(
                    "must be at least m ({}) for usable recall, got {}",
                    self.m, self.ef_construction
                ),
            });
        }
        if let Some(m0) = self.m0 {
            if m0 < self.m {
                return Err(Error::InvalidConfig {
                    parameter: "m0",
                    reason: format!("must be at least m ({}), got {m0}", self.m),
                });
            }
        }
        Ok(())
    }

    #[inline] pub(crate) fn m0(&self) -> usize { self.m0.unwrap_or(2 * self.m) }
    #[inline] pub(crate) fn max_links(&self, layer: usize) -> usize {
        if layer == 0 { self.m0() } else { self.m }
    }
    #[inline] pub(crate) fn m_l(&self) -> f64 { 1.0 / (self.m as f64).ln() }
}

// ─── Search result ────────────────────────────────────────────────────────────

/// One result returned by [`Hnsw::search`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub id:       usize,
    pub distance: f32,
}

// ─── Main index ───────────────────────────────────────────────────────────────

/// Hierarchical Navigable Small World approximate nearest-neighbour index.
///
/// # Type parameter
/// * `D` – a [`Distance`] implementation (e.g. [`crate::distance::Euclidean`]).
///
/// # Example
/// ```
/// use fast_hnsw::{Hnsw, Config};
/// use fast_hnsw::distance::Euclidean;
///
/// let mut index = Hnsw::new(Config::default(), Euclidean)?;
/// index.insert(vec![1.0, 0.0]).unwrap();
/// index.insert(vec![0.0, 1.0]).unwrap();
/// index.insert(vec![0.5, 0.5]).unwrap();
///
/// let results = index.search(&[0.9, 0.1], 1, 20)?;
/// assert_eq!(results[0].id, 0); // [1,0] is closest to [0.9,0.1]
/// # Ok::<(), fast_hnsw::Error>(())
/// ```
pub struct Hnsw<D: Distance> {
    pub(crate) config: Config,
    pub(crate) metric: D,
    /// Flat vector store: vector `i` at `data[i*dim .. (i+1)*dim]`.
    pub(crate) vec_store: VecStore,
    /// HNSW adjacency storage, addressed by node and layer.
    ///
    /// Storing the distance alongside the id enables the heuristic reverse-update
    /// prune to skip all M distance recomputations — only the O(M²/2) pairwise
    /// diversity checks remain.
    pub(crate) graph: GraphStore,
    pub(crate) entry_point: Option<(usize, usize)>,
    rng:         SmallRng,
    pub(crate) dim: Option<usize>,
    visited:     VisitedTracker,
    scratch:     Scratch,
    ep_buf:      Vec<DistId>,
    /// Output buffer for `select_neighbours_*` — reused across calls.
    select_buf:  Vec<(usize, f32)>,
    /// Discarded-candidates buffer for `keep_pruned` path — reused.
    pruned_buf:  Vec<(usize, f32)>,
    /// Sorted candidate buffer for `prune_connections_heuristic` — reused.
    prune_buf:   Vec<(usize, f32)>,
    /// Selected bidirectional edges retained across insertion layers.
    edge_buf:    Vec<Edge>,
    /// Tombstones, indexed by node id.
    ///
    /// Left empty until the first [`Hnsw::remove`], so an index that never
    /// deletes pays nothing for the feature.  Deleted nodes stay in the graph
    /// as navigation waypoints — removing their edges would disconnect the
    /// small-world structure — and are filtered out of results instead.
    deleted: Vec<bool>,
    /// Number of `true` entries in `deleted`, so `live_len` stays O(1).
    deleted_count: usize,
}

impl<D: Distance> Hnsw<D> {
    // ─── Construction ─────────────────────────────────────────────────────

    /// Create a new, empty index.
    pub fn new(config: Config, metric: D) -> Result<Self> {
        config.validate()?;
        let cap = config.capacity;
        let ef  = config.ef_construction;
        let m   = config.m;
        Ok(Self {
            config,
            metric,
            vec_store:   VecStore::new(0, cap),
            graph:        GraphStore::with_capacity(cap),
            entry_point: None,
            rng:         rand::make_rng(),
            dim:         None,
            visited:     VisitedTracker::new(cap.max(64)),
            scratch:     Scratch::new(ef),
            ep_buf:      Vec::with_capacity(ef),
            select_buf:  Vec::with_capacity(m * 2 + 2),
            pruned_buf:  Vec::with_capacity(m * 2 + 2),
            prune_buf:   Vec::with_capacity(m * 2 + 2),
            edge_buf:    Vec::with_capacity(m * 2 + 2),
            deleted:     Vec::new(),
            deleted_count: 0,
        })
    }

    /// Reconstruct an index from its already-deserialized components.
    ///
    /// Called exclusively by the persistence layer (`persist::read_hnsw`).
    /// All the heavy lifting (reading header, vectors, graph) is done by the
    /// caller; this just wires everything into the struct.
    pub(crate) fn from_parts(
        config:      Config,
        metric:      D,
        vec_store:   VecStore,
        graph:       GraphStore,
        entry_point: Option<(usize, usize)>,
        dim:         Option<usize>,
    ) -> Self {
        let n  = vec_store.len();
        let ef = config.ef_construction;
        let m  = config.m;
        Self {
            config,
            metric,
            vec_store,
            graph,
            entry_point,
            rng:        rand::make_rng(),
            dim,
            visited:    VisitedTracker::new(n.max(64)),
            scratch:    Scratch::new(ef),
            ep_buf:     Vec::with_capacity(ef),
            select_buf: Vec::with_capacity(m * 2 + 2),
            pruned_buf: Vec::with_capacity(m * 2 + 2),
            prune_buf:  Vec::with_capacity(m * 2 + 2),
            edge_buf:   Vec::with_capacity(m * 2 + 2),
            deleted:    Vec::new(),
            deleted_count: 0,
        }
    }

    /// Create a new index with a fixed RNG seed (reproducible for tests).
    pub fn new_with_seed(config: Config, metric: D, seed: u64) -> Result<Self> {
        config.validate()?;
        let cap = config.capacity;
        let ef  = config.ef_construction;
        let m   = config.m;
        Ok(Self {
            config,
            metric,
            vec_store:   VecStore::new(0, cap),
            graph:        GraphStore::with_capacity(cap),
            entry_point: None,
            rng:         SmallRng::seed_from_u64(seed),
            dim:         None,
            visited:     VisitedTracker::new(cap.max(64)),
            scratch:     Scratch::new(ef),
            ep_buf:      Vec::with_capacity(ef),
            select_buf:  Vec::with_capacity(m * 2 + 2),
            pruned_buf:  Vec::with_capacity(m * 2 + 2),
            prune_buf:   Vec::with_capacity(m * 2 + 2),
            edge_buf:    Vec::with_capacity(m * 2 + 2),
            deleted:     Vec::new(),
            deleted_count: 0,
        })
    }

    // ─── Public API ───────────────────────────────────────────────────────

    /// Insert a vector and return its assigned id (0-based).
    ///
    /// # Panics
    /// Panics if `vector.len()` differs from the dimension of previously
    /// inserted vectors.
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<usize> {
        let dim = vector.len();
        match self.dim {
            None => {
                if self.vec_store.mmap.is_some() {
                    return Err(Error::ReadOnly);
                }
                self.dim = Some(dim);
                self.vec_store.dim = dim;
            }
            Some(d) => {
                if d != dim {
                    return Err(Error::DimensionMismatch { expected: d, actual: dim });
                }
                if self.vec_store.mmap.is_some() {
                    return Err(Error::ReadOnly);
                }
            }
        }

        let q       = self.vec_store.len();
        let q_level = self.random_level();

        self.vec_store.push(vector);

        // Pre-allocate connection list with inner Vecs sized to capacity.
        let mut conn: Vec<Vec<(u32, f32)>> = Vec::with_capacity(q_level + 1);
        for l in 0..=q_level {
            conn.push(Vec::with_capacity(self.config.max_links(l)));
        }
        self.graph.push_node(conn);

        if self.visited.stamps.len() <= q {
            self.visited.stamps.resize(q * 2 + 1, 0);
        }

        // ── First insertion ───────────────────────────────────────────────
        let (ep_id, ep_level) = match self.entry_point {
            None => { self.entry_point = Some((q, q_level)); return Ok(q); }
            Some(x) => x,
        };

        // ── Phase 1: greedy descent from ep_level down to q_level+1 ──────
        self.ep_buf.clear();
        self.ep_buf.push(DistId::new(self.dist(q, ep_id), ep_id));

        for layer in (q_level + 1..=ep_level).rev() {
            self.search_layer_node(q, 1, layer);
            std::mem::swap(&mut self.ep_buf, &mut self.scratch.out);
        }

        // ── Phase 2: insert q at layers q_level..=0 ──────────────────────
        let top = ep_level.min(q_level);
        for layer in (0..=top).rev() {
            let ef    = self.config.ef_construction;
            let m_max = self.config.max_links(layer);

            self.search_layer_node(q, ef, layer);

            // Select M neighbours for q.  Results written to `self.select_buf`.
            if self.config.use_heuristic {
                self.select_neighbours_heuristic(q, m_max, layer);
            } else {
                self.select_neighbours_simple(m_max);
            }

            // Add bidirectional edges.
            //
            // We copy select_buf into a retained edge buffer because
            // `prune_connections_heuristic` (called below) overwrites select_buf
            // and pruned_buf when it runs its own heuristic selection.
            let mut edge_buf = std::mem::take(&mut self.edge_buf);
            edge_buf.clear();
            edge_buf.extend(
                self.select_buf
                    .iter()
                    .map(|&(neighbour, distance)| (neighbour as u32, distance)),
            );

            // Pass 1: add all edges (keeps the connection lists coherent before
            // any pruning modifies them).
            //
            // q's own list is resolved once instead of per edge; the reverse
            // edges each touch a different node, so they still need a lookup
            // apiece.  Splitting the two directions does not change either
            // list's final contents or ordering.
            {
                let q_links = self.graph.neighbours_mut(q, layer);
                q_links.extend(edge_buf.iter().copied());
            }
            for &(nb_u32, dist_q_nb) in &edge_buf {
                self.graph
                    .neighbours_mut(nb_u32 as usize, layer)
                    .push((q as u32, dist_q_nb));
            }

            // Pass 2: prune any neighbour whose list now exceeds m_max.
            //
            // Which strategy fires is determined by `self.config.prune_strategy`
            // (set via `Builder::prune_strategy`).  Both branches use the `f32`
            // distance that is stored alongside every neighbour id — so neither
            // branch needs to recompute the M distances from scratch.
            for &(nb_u32, _) in &edge_buf {
                let nb = nb_u32 as usize;
                if self.graph.neighbour_count(nb, layer) > m_max {
                    match self.config.prune_strategy {

                        // ── Simple: sort stored distances + truncate ──────────
                        // Keeps the M nearest neighbours by raw distance.
                        // Cost: O(M log M) sort of f32 values already in the
                        //       connection list — no vector data touched, no new
                        //       distance computation.  ~25 ns per call.
                        PruneStrategy::Simple => {
                            self.graph.neighbours_mut(nb, layer)
                                .sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                            self.graph.neighbours_mut(nb, layer).truncate(m_max);
                        }

                        // ── Heuristic: full Algorithm 4 with stored distances ─
                        // Runs the diversity check: keeps candidate `e` only if
                        // d(nb, e) ≤ d(e, s) for every already-selected s.
                        // d(nb, e) is the stored f32 — zero recomputation.
                        // Only the O(M²/2) pairwise d(e, s) checks are computed,
                        // and ~60% are skipped by the triangle-inequality shortcut.
                        // ~1–11 µs per call depending on cache state.
                        PruneStrategy::Heuristic => {
                            self.prune_connections_heuristic(nb, layer, m_max);
                        }
                    }
                }
            }
            self.edge_buf = edge_buf;

            std::mem::swap(&mut self.ep_buf, &mut self.scratch.out);
        }

        if q_level > ep_level {
            self.entry_point = Some((q, q_level));
        }
        Ok(q)
    }

    /// Search for the `k` approximate nearest neighbours of `query`.
    ///
    /// `ef` controls recall vs. speed (`ef ≥ k`; larger → better recall).
    ///
    /// Repeated calls on one thread reuse that thread's traversal storage, so
    /// the per-query cost does not scale with the size of the index.  Use
    /// [`Hnsw::search_with_workspace`] to own that storage explicitly.
    ///
    /// # Panics
    /// Panics if `query.len()` differs from the index dimension, or if `k` is
    /// zero.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>> {
        with_query_workspace(|workspace| self.search_with_workspace(query, k, ef, workspace))
    }

    /// Search using caller-owned storage retained across queries.
    ///
    /// Results are identical to [`Hnsw::search`], but after the workspace has
    /// reached the required node and `ef` capacities, traversal performs no
    /// visited-set or heap-buffer allocation. The returned result vector is
    /// still owned by the caller.
    ///
    /// # Panics
    /// Panics if `query.len()` differs from the index dimension, or if `k` is
    /// zero.
    pub fn search_with_workspace(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        workspace: &mut SearchWorkspace,
    ) -> Result<Vec<SearchResult>> {
        self.try_search_with_workspace_map(
            query,
            k,
            ef,
            workspace,
            Ok,
        )
    }

    /// Search and transform each result directly into the caller's output type.
    ///
    /// Traversal still reuses [`SearchWorkspace`], while the returned vector is
    /// allocated once for `T`. This avoids first materializing an intermediate
    /// `Vec<SearchResult>` in adapters that immediately decode or convert every
    /// result. A transform error stops conversion without invalidating the
    /// reusable workspace.
    pub fn try_search_with_workspace_map<T, E, M>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        workspace: &mut SearchWorkspace,
        map: M,
    ) -> std::result::Result<Vec<T>, E>
    where
        E: From<Error>,
        M: FnMut(SearchResult) -> std::result::Result<T, E>,
    {
        self.check_query_dim(query).map_err(E::from)?;
        self.try_search_with_distance_and_workspace_map(
            k,
            ef,
            |id| self.metric.distance(query, self.vec_store.get(id)),
            workspace,
            map,
        )
    }

    /// Search using a query-prepared external distance function.
    ///
    /// The callback receives a zero-based node id and returns its distance to
    /// the current query. This lets read-only callers traverse the stored graph
    /// while scoring vectors held in another representation (for example a
    /// compressed mmap) without coupling HNSW to that storage format.
    pub fn search_with_distance<F>(
        &self,
        k: usize,
        ef: usize,
        distance: F,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(usize) -> f32,
    {
        with_query_workspace(|workspace| {
            self.search_with_distance_and_workspace(k, ef, &distance, workspace)
        })
    }

    /// External-distance search with caller-owned reusable workspace.
    pub fn search_with_distance_and_workspace<F>(
        &self,
        k: usize,
        ef: usize,
        distance: F,
        workspace: &mut SearchWorkspace,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(usize) -> f32,
    {
        self.try_search_with_distance_and_workspace_map(k, ef, distance, workspace, Ok)
    }

    /// External-distance search that transforms workspace results directly.
    pub fn try_search_with_distance_and_workspace_map<T, E, F, M>(
        &self,
        k: usize,
        ef: usize,
        distance: F,
        workspace: &mut SearchWorkspace,
        mut map: M,
    ) -> std::result::Result<Vec<T>, E>
    where
        E: From<Error>,
        F: Fn(usize) -> f32,
        M: FnMut(SearchResult) -> std::result::Result<T, E>,
    {
        // Tombstoned nodes must stay navigable but never surface as results,
        // which is exactly the filtered traversal's semantics.  Routing through
        // it keeps the deletion check in one place; indexes with no deletions
        // keep using the cheaper unfiltered path below.
        if self.deleted_count > 0 {
            return self.try_search_filtered_with_distance_and_workspace_map(
                k,
                ef,
                distance,
                |_| true,
                workspace,
                map,
            );
        }

        if k == 0 {
            return Err(E::from(Error::ZeroK));
        }
        let ef = ef.max(k);

        let (ep_id, ep_level) = match self.entry_point {
            None    => return Ok(Vec::new()),
            Some(x) => x,
        };

        workspace.prepare(self.vec_store.len(), ef);
        let SearchWorkspace {
            visited,
            scratch,
            entry_points,
        } = workspace;

        let ep_dist = distance(ep_id);
        entry_points.push(DistId::new(ep_dist, ep_id));

        for layer in (1..=ep_level).rev() {
            Self::do_search_layer_with_distance(
                &self.graph,
                visited,
                scratch,
                entry_points,
                1,
                layer,
                &distance,
            );
            std::mem::swap(entry_points, &mut scratch.out);
        }

        Self::do_search_layer_with_distance(
            &self.graph,
            visited,
            scratch,
            entry_points,
            ef,
            0,
            &distance,
        );
        scratch.out.truncate(k);
        let mut output = Vec::with_capacity(scratch.out.len());
        for result in &scratch.out {
            output.push(map(SearchResult {
                id: result.id,
                distance: result.dist,
            })?);
        }
        Ok(output)
    }

    /// Search while applying an eligibility predicate during layer-0
    /// traversal.
    ///
    /// Rejected nodes remain navigation candidates, preserving connectivity
    /// through mixed or highly selective graphs, but they never enter the
    /// bounded result heap. Consequently `k` is applied to accepted nodes
    /// rather than to an unfiltered top-k followed by post-filtering.
    ///
    /// The predicate receives the zero-based vector id. If fewer than `k`
    /// accepted nodes are reachable, all accepted results found are returned.
    ///
    /// # Panics
    /// Panics if `query.len()` differs from the index dimension, or if `k` is
    /// zero.
    pub fn search_filtered<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accepts: F,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(usize) -> bool,
    {
        with_query_workspace(|workspace| {
            self.search_filtered_with_workspace(query, k, ef, &accepts, workspace)
        })
    }

    /// Filtered search using caller-owned storage retained across queries.
    ///
    /// This combines the filter-before-top-k semantics of
    /// [`Hnsw::search_filtered`] with the allocation reuse of
    /// [`Hnsw::search_with_workspace`].
    ///
    /// # Panics
    /// Panics if `query.len()` differs from the index dimension, or if `k` is
    /// zero.
    pub fn search_filtered_with_workspace<F>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accepts: F,
        workspace: &mut SearchWorkspace,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(usize) -> bool,
    {
        self.try_search_filtered_with_workspace_map(
            query,
            k,
            ef,
            accepts,
            workspace,
            Ok,
        )
    }

    /// Filtered search that transforms workspace results directly.
    pub fn try_search_filtered_with_workspace_map<T, E, F, M>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accepts: F,
        workspace: &mut SearchWorkspace,
        map: M,
    ) -> std::result::Result<Vec<T>, E>
    where
        E: From<Error>,
        F: Fn(usize) -> bool,
        M: FnMut(SearchResult) -> std::result::Result<T, E>,
    {
        self.check_query_dim(query).map_err(E::from)?;
        self.try_search_filtered_with_distance_and_workspace_map(
            k,
            ef,
            |id| self.metric.distance(query, self.vec_store.get(id)),
            accepts,
            workspace,
            map,
        )
    }

    /// Filtered graph traversal using an external distance-by-node function.
    pub fn search_filtered_with_distance<F, A>(
        &self,
        k: usize,
        ef: usize,
        distance: F,
        accepts: A,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(usize) -> f32,
        A: Fn(usize) -> bool,
    {
        with_query_workspace(|workspace| {
            self.search_filtered_with_distance_and_workspace(
                k,
                ef,
                &distance,
                &accepts,
                workspace,
            )
        })
    }

    /// Filtered external-distance search with caller-owned reusable workspace.
    pub fn search_filtered_with_distance_and_workspace<F, A>(
        &self,
        k: usize,
        ef: usize,
        distance: F,
        accepts: A,
        workspace: &mut SearchWorkspace,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(usize) -> f32,
        A: Fn(usize) -> bool,
    {
        self.try_search_filtered_with_distance_and_workspace_map(
            k, ef, distance, accepts, workspace, Ok,
        )
    }

    /// Filtered external-distance search that transforms workspace results directly.
    pub fn try_search_filtered_with_distance_and_workspace_map<T, E, F, A, M>(
        &self,
        k: usize,
        ef: usize,
        distance: F,
        accepts: A,
        workspace: &mut SearchWorkspace,
        mut map: M,
    ) -> std::result::Result<Vec<T>, E>
    where
        E: From<Error>,
        F: Fn(usize) -> f32,
        A: Fn(usize) -> bool,
        M: FnMut(SearchResult) -> std::result::Result<T, E>,
    {
        if k == 0 {
            return Err(E::from(Error::ZeroK));
        }
        let ef = ef.max(k);

        let (ep_id, ep_level) = match self.entry_point {
            None => return Ok(Vec::new()),
            Some(entry_point) => entry_point,
        };

        workspace.prepare(self.vec_store.len(), ef);
        let SearchWorkspace {
            visited,
            scratch,
            entry_points,
        } = workspace;

        let entry_distance = distance(ep_id);
        entry_points.push(DistId::new(entry_distance, ep_id));

        // Upper layers are navigation-only and deliberately ignore the
        // eligibility predicate.
        for layer in (1..=ep_level).rev() {
            Self::do_search_layer_with_distance(
                &self.graph,
                visited,
                scratch,
                entry_points,
                1,
                layer,
                &distance,
            );
            std::mem::swap(entry_points, &mut scratch.out);
        }

        // Tombstoned nodes are excluded here rather than by the caller, so
        // every search entry point — plain, filtered, and external-distance —
        // honours deletions through this one predicate.  They still relay
        // traversal, because the filtered layer search keeps rejected nodes as
        // navigation candidates.
        let eligible = |id: usize| !self.is_deleted(id) && accepts(id);

        Self::do_search_layer_filtered_with_distance(
            &self.graph,
            visited,
            scratch,
            entry_points,
            ef,
            &eligible,
            &distance,
        );
        scratch.out.truncate(k);
        let mut output = Vec::with_capacity(scratch.out.len());
        for result in &scratch.out {
            output.push(map(SearchResult {
                id: result.id,
                distance: result.dist,
            })?);
        }
        Ok(output)
    }

    // ─── Exact search ─────────────────────────────────────────────────────

    /// Exhaustive k-nearest-neighbour search: scores every live vector.
    ///
    /// Unlike [`Hnsw::search`] this is exact, at `O(n · dim)` per query. It
    /// exists for the cases where that is what you want — computing ground
    /// truth to measure the approximate search's recall, and small indexes
    /// where a full scan beats a graph traversal. Tombstoned vectors are
    /// skipped, as in every other search.
    ///
    /// With the `blas` feature and an inner-product metric this dispatches to
    /// a single `sgemv` over the whole vector store, which is where BLAS
    /// genuinely wins: one call amortised across every candidate, rather than
    /// the per-vector call that makes it lose in graph traversal. Measured
    /// 4.5–24.8× over a scalar loop depending on dimension and batch size.
    /// Every other metric uses the vectorised kernels.
    ///
    /// # Panics
    /// Panics if `query.len()` differs from the index dimension, if `k` is
    /// zero, or if the vectors are not resident (a `pread` index — read them
    /// through [`PreadIndex`](crate::pread::PreadIndex) instead).
    pub fn exact_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.check_query_dim(query)?;
        if k == 0 {
            return Err(Error::ZeroK);
        }
        if self.vec_store.deferred_len().is_some() {
            return Err(Error::VectorsNotResident);
        }
        let count = self.vec_store.len();
        if count == 0 {
            return Ok(Vec::new());
        }

        // BLAS fast path: only for the inner-product metric, where a matrix
        // multiply computes precisely what the metric is defined as. The L2
        // metrics could be expressed through the norm expansion, but that
        // loses most of its significant digits for near-identical vectors —
        // exactly the ones a k-NN query must rank correctly. See the
        // `distance` module.
        #[cfg(feature = "blas")]
        {
            if D::metric_id() == crate::distance::DotProduct::metric_id() {
                if let Some(scores) = self.gemv_inner_products(query) {
                    return Ok(Self::top_k(
                        k,
                        count,
                        |id| 1.0 - scores[id],
                        |id| self.is_deleted(id),
                    ));
                }
            }
        }

        Ok(Self::top_k(
            k,
            count,
            |id| self.metric.distance(query, self.vec_store.get(id)),
            |id| self.is_deleted(id),
        ))
    }

    /// All inner products between `query` and the vector store, via one
    /// `sgemv`. `None` when the vectors are not contiguous.
    #[cfg(feature = "blas")]
    fn gemv_inner_products(&self, query: &[f32]) -> Option<Vec<f32>> {
        let vectors = self.vec_store.as_slice()?;
        let dim = self.dim?;
        let count = self.vec_store.len();
        if dim == 0 || count == 0 || count > i32::MAX as usize || dim > i32::MAX as usize {
            return None;
        }
        let mut scores = vec![0.0f32; count];
        // SAFETY: `vectors` holds `count * dim` contiguous floats (checked by
        // `VecStore`), `query` holds `dim` (checked by `assert_query_dim`), and
        // `scores` has room for `count`. The casts are guarded above.
        unsafe {
            crate::blas::cblas_sgemv(
                crate::blas::CBLAS_ROW_MAJOR,
                crate::blas::CBLAS_NO_TRANS,
                count as i32,
                dim as i32,
                1.0,
                vectors.as_ptr(),
                dim as i32,
                query.as_ptr(),
                1,
                0.0,
                scores.as_mut_ptr(),
                1,
            );
        }
        Some(scores)
    }

    /// Bounded top-k over `count` ids, skipping those `skip` rejects.
    fn top_k(
        k: usize,
        count: usize,
        distance: impl Fn(usize) -> f32,
        skip: impl Fn(usize) -> bool,
    ) -> Vec<SearchResult> {
        let mut worst = BinaryHeap::with_capacity(k + 1);
        for id in 0..count {
            if skip(id) {
                continue;
            }
            worst.push(DistId::new(distance(id), id));
            if worst.len() > k {
                worst.pop();
            }
        }
        worst
            .into_sorted_vec()
            .into_iter()
            .map(|entry| SearchResult { id: entry.id, distance: entry.dist })
            .collect()
    }

    // ─── Deletion ─────────────────────────────────────────────────────────

    /// Mark `id` as deleted.  Returns `false` if it was already deleted or is
    /// out of range.
    ///
    /// This is a **soft delete**.  The node keeps its place in the graph and
    /// still relays traversals, but it is excluded from every search result.
    /// Physically removing it would sever the edges that make neighbouring
    /// nodes reachable, so HNSW implementations universally tombstone instead.
    ///
    /// Ids of surviving vectors never shift, which is what lets a
    /// [`LabeledIndex`](crate::labeled::LabeledIndex) or
    /// [`PairedIndex`](crate::paired::PairedIndex) keep addressing payloads by
    /// the same id.  Consequently [`Hnsw::len`] still counts tombstoned slots;
    /// use [`Hnsw::live_len`] for the number of reachable vectors.
    ///
    /// Deleting works on a memory-mapped index too — the tombstones live in
    /// memory and never write to the mapping.
    ///
    /// Recall degrades once tombstones dominate, because searches spend their
    /// `ef` budget traversing dead nodes.  Rebuild with
    /// [`Hnsw::compacted`] when [`Hnsw::deleted_count`] grows large.
    pub fn remove(&mut self, id: usize) -> bool {
        if id >= self.vec_store.len() {
            return false;
        }
        if self.deleted.is_empty() {
            self.deleted = vec![false; self.vec_store.len()];
        } else if self.deleted.len() < self.vec_store.len() {
            self.deleted.resize(self.vec_store.len(), false);
        }
        if self.deleted[id] {
            return false;
        }
        self.deleted[id] = true;
        self.deleted_count += 1;
        true
    }

    /// Clear the tombstone on `id`, making it visible to searches again.
    /// Returns `false` if it was not deleted.
    pub fn restore(&mut self, id: usize) -> bool {
        if id >= self.deleted.len() || !self.deleted[id] {
            return false;
        }
        self.deleted[id] = false;
        self.deleted_count -= 1;
        true
    }

    /// Whether `id` has been tombstoned.
    #[inline]
    pub fn is_deleted(&self, id: usize) -> bool {
        // The common case is an index with no deletions at all, where the
        // vector is empty and this is a single length check.
        !self.deleted.is_empty() && self.deleted.get(id).copied().unwrap_or(false)
    }

    /// Number of tombstoned slots.
    #[inline]
    pub fn deleted_count(&self) -> usize { self.deleted_count }

    /// Number of vectors still reachable by search — [`Hnsw::len`] minus
    /// [`Hnsw::deleted_count`].
    #[inline]
    pub fn live_len(&self) -> usize { self.vec_store.len() - self.deleted_count }

    /// Rebuild into a fresh index holding only the live vectors.
    ///
    /// Returns the new index together with a map from **new id to old id**, so
    /// callers can carry side tables across the renumbering.  Tombstones are
    /// discarded; ids are densely reassigned in ascending order of the old
    /// ids, so the map is sorted.
    ///
    /// The graph is rebuilt by reinsertion, which costs about as much as
    /// building the original index. Pass `seed` to make the rebuild
    /// reproducible.
    ///
    /// This is also how you persist an index that has deletions: [`save`] and
    /// its variants refuse a tombstoned index, because the on-disk format has
    /// no place to record tombstones and silently writing the deleted vectors
    /// back would resurrect them on load.
    ///
    /// [`save`]: crate::persist::save
    pub fn compacted(&self, metric: D, seed: Option<u64>) -> Result<(Hnsw<D>, Vec<usize>)> {
        let mut config = self.config.clone();
        config.capacity = self.live_len();
        let mut rebuilt = match seed {
            Some(seed) => Hnsw::new_with_seed(config, metric, seed)?,
            None => Hnsw::new(config, metric)?,
        };
        let mut old_ids = Vec::with_capacity(self.live_len());
        for old_id in 0..self.vec_store.len() {
            if self.is_deleted(old_id) {
                continue;
            }
            rebuilt.insert(self.vec_store.get(old_id).to_vec())?;
            old_ids.push(old_id);
        }
        Ok((rebuilt, old_ids))
    }

    #[inline] pub fn len(&self)              -> usize         { self.vec_store.len() }
    #[inline] pub fn is_empty(&self)         -> bool          { self.vec_store.len() == 0 }
    /// The stored vector for `id`.
    #[inline]
    pub fn get_vector(&self, id: usize) -> Result<&[f32]> {
        let len = self.vec_store.len();
        if id >= len {
            return Err(Error::IdOutOfBounds { id, len });
        }
        if self.vec_store.deferred_len().is_some() {
            return Err(Error::VectorsNotResident);
        }
        Ok(self.vec_store.get(id))
    }
    #[inline] pub fn dim(&self)              -> Option<usize> { self.dim }
    #[inline] pub fn config(&self)           -> &Config       { &self.config }
    /// The distance metric this index was built with.
    #[inline] pub fn metric(&self)           -> &D            { &self.metric }
    pub fn max_level(&self) -> Option<usize> { self.entry_point.map(|(_, l)| l) }

    // ─── Level generation ─────────────────────────────────────────────────

    fn random_level(&mut self) -> usize {
        let u: f64 = self.rng.random::<f64>().max(f64::MIN_POSITIVE);
        (-u.ln() * self.config.m_l()).floor() as usize
    }

    // ─── Query validation ─────────────────────────────────────────────────

    /// Reject a query whose length does not match the indexed dimension.
    ///
    /// An empty index has no dimension yet and accepts anything.
    ///
    /// Every built-in metric folds with `zip` semantics, so a mismatched query
    /// would otherwise be silently truncated (or silently ignore its extra
    /// components) and return confident, wrong neighbours.  An empty index has
    /// no dimension yet and accepts anything — it returns no results regardless.
    #[inline]
    pub(crate) fn check_query_dim(&self, query: &[f32]) -> Result<()> {
        match self.dim {
            Some(dim) if query.len() != dim => Err(Error::DimensionMismatch {
                expected: dim,
                actual: query.len(),
            }),
            _ => Ok(()),
        }
    }

    // ─── Distance helpers ─────────────────────────────────────────────────

    #[inline]
    fn dist(&self, a: usize, b: usize) -> f32 {
        self.metric.distance(self.vec_store.get(a), self.vec_store.get(b))
    }

    // ─── search_layer (insert path — uses self.scratch) ───────────────────

    fn search_layer_node(&mut self, q: usize, ef: usize, layer: usize) {
        let vec_store   = &self.vec_store;
        let graph       = &self.graph;
        let metric      = &self.metric;
        let visited     = &mut self.visited;
        let scratch     = &mut self.scratch;
        let ep          = &self.ep_buf;

        let q_vec = vec_store.get(q);
        visited.begin();
        scratch.begin(ef);

        for &ep_d in ep {
            if visited.visit(ep_d.id) { scratch.push_candidate(ep_d); }
        }

        while let Some(c) = scratch.pop_candidate() {
            let Some(worst) = scratch.worst_result_dist() else { break };
            if c.dist > worst { break; }

            if let Some(nb_list) = graph.neighbours(c.id, layer) {
                for (nb_u32, _) in nb_list {
                    let nb = nb_u32 as usize;
                    if visited.visit(nb) {
                        let nb_dist = metric.distance(q_vec, vec_store.get(nb));
                        let cur_worst = scratch.worst_result_dist().unwrap_or(f32::INFINITY);
                        if nb_dist < cur_worst || scratch.results_len() < ef {
                            scratch.push_candidate(DistId::new(nb_dist, nb));
                        }
                    }
                }
            }
        }
        scratch.finish();
    }

    // ─── search_layer (search path — takes explicit params) ───────────────

    fn do_search_layer_with_distance<F>(
        graph: &GraphStore,
        visited: &mut VisitedTracker,
        scratch: &mut Scratch,
        entry_points: &[DistId],
        ef: usize,
        layer: usize,
        distance: &F,
    ) where
        F: Fn(usize) -> f32,
    {
        visited.begin();
        scratch.begin(ef);

        for &ep in entry_points {
            if visited.visit(ep.id) { scratch.push_candidate(ep); }
        }

        while let Some(c) = scratch.pop_candidate() {
            let Some(worst) = scratch.worst_result_dist() else { break };
            if c.dist > worst { break; }

            if let Some(nb_list) = graph.neighbours(c.id, layer) {
                for (nb_u32, _) in nb_list {
                    let nb = nb_u32 as usize;
                    if visited.visit(nb) {
                        let nb_dist = distance(nb);
                        let cur_worst = scratch.worst_result_dist().unwrap_or(f32::INFINITY);
                        if nb_dist < cur_worst || scratch.results_len() < ef {
                            scratch.push_candidate(DistId::new(nb_dist, nb));
                        }
                    }
                }
            }
        }
        scratch.finish();
    }

    // ─── search_layer (filtered layer-0 search) ───────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn do_search_layer_filtered_with_distance<F, A>(
        graph: &GraphStore,
        visited: &mut VisitedTracker,
        scratch: &mut Scratch,
        entry_points: &[DistId],
        ef: usize,
        accepts: &A,
        distance: &F,
    ) where
        F: Fn(usize) -> f32,
        A: Fn(usize) -> bool,
    {
        visited.begin();
        scratch.begin(ef);

        for &entry in entry_points {
            if visited.visit(entry.id) {
                scratch.push_navigation(entry);
                if accepts(entry.id) {
                    scratch.push_result(entry);
                }
            }
        }

        while let Some(candidate) = scratch.pop_candidate() {
            let worst = scratch.worst_result_dist().unwrap_or(f32::INFINITY);
            if scratch.results_len() >= ef && candidate.dist > worst {
                break;
            }

            if let Some(neighbours) = graph.neighbours(candidate.id, 0) {
                for (neighbour_id, _) in neighbours {
                    let neighbour = neighbour_id as usize;
                    if !visited.visit(neighbour) {
                        continue;
                    }

                    let distance = distance(neighbour);
                    let worst = scratch.worst_result_dist().unwrap_or(f32::INFINITY);
                    if scratch.results_len() < ef || distance < worst {
                        let neighbour = DistId::new(distance, neighbour);
                        scratch.push_navigation(neighbour);
                        if accepts(neighbour.id) {
                            scratch.push_result(neighbour);
                        }
                    }
                }
            }
        }
        scratch.finish();
    }

    // ─── Neighbour selection ──────────────────────────────────────────────

    /// **Algorithm 3** – write the `m` closest entries from `scratch.out`
    /// (sorted closest-first) into `self.select_buf`.
    fn select_neighbours_simple(&mut self, m: usize) {
        self.select_buf.clear();
        let end = m.min(self.scratch.out.len());
        // scratch.out is Copy, access by index is safe with any mutable alias of select_buf
        for i in 0..end {
            let d = self.scratch.out[i];
            self.select_buf.push((d.id, d.dist));
        }
    }

    /// **Algorithm 4 (heuristic)** – write up to `m` diverse neighbours into
    /// `self.select_buf`.  Results are `(node_id, dist_from_q)`.
    ///
    /// Key optimisations vs. the previous version:
    /// * **No heap rebuild** – `scratch.out` is already sorted closest-first;
    ///   we iterate it directly in O(n) instead of rebuilding a min-heap in
    ///   O(n log n) + 1 allocation.
    /// * **Early return** – when `|candidates| ≤ m`, all candidates are
    ///   automatically selected (no pairwise diversity check needed).
    /// * **Triangle-inequality shortcut** – if `d(q,s) > 2·d(q,e)`, skip the
    ///   actual `d(e,s)` computation.
    /// * **`select_buf` / `pruned_buf` reuse** – no per-call allocation.
    fn select_neighbours_heuristic(&mut self, q: usize, m: usize, layer: usize) {
        self.select_buf.clear();
        self.pruned_buf.clear();

        // ── Fast path: fewer candidates than slots → take all ────────────
        let n_cands = self.scratch.out.len();
        if n_cands <= m && !self.config.extend_candidates {
            for i in 0..n_cands {
                let d = self.scratch.out[i]; // Copy
                self.select_buf.push((d.id, d.dist));
            }
            return;
        }

        // ── Extended-candidates path (rare, allocates a temporary) ────────
        // Build an extended candidate list that includes the neighbours-of-
        // candidates.  Only triggered when extend_candidates = true.
        let ext_buf: Vec<DistId>;
        let cands: &[DistId] = if self.config.extend_candidates {
            let mut tmp: Vec<DistId> = self.scratch.out.to_vec();
            let seen_ids: std::collections::HashSet<usize> =
                self.scratch.out.iter().map(|d| d.id).collect();
            let mut extra: Vec<DistId> = Vec::new();
            for &d in &self.scratch.out {
                if let Some(nb_list) = self.graph.neighbours(d.id, layer) {
                    for (nb_u32, _) in nb_list {
                        let nb = nb_u32 as usize;
                        if !seen_ids.contains(&nb) {
                            extra.push(DistId::new(self.dist(q, nb), nb));
                        }
                    }
                }
            }
            tmp.extend_from_slice(&extra);
            tmp.sort_unstable_by(|a, b| a.dist.total_cmp(&b.dist));
            ext_buf = tmp;
            &ext_buf
        } else {
            &self.scratch.out
        };

        // ── Main heuristic loop ───────────────────────────────────────────
        //
        // Iterate candidates in closest-first order.  Accept candidate `e` iff
        // `d(q, e) ≤ d(e, s)` for every already-accepted neighbour `s`.
        // Equivalently, reject if any `s` is closer to `e` than `q` is.
        for candidate in cands {
            if self.select_buf.len() >= m { break; }
            let e_dist = candidate.dist;
            let e_id   = candidate.id;

            let mut accept = true;
            for j in 0..self.select_buf.len() {
                let (s_id, s_dist_q) = self.select_buf[j]; // Copy
                // Triangle-inequality shortcut:
                // If d(q,s) > 2·d(q,e) → d(e,s) ≥ d(q,s)−d(q,e) > d(q,e)
                // so the condition d(q,e) ≤ d(e,s) is guaranteed → continue
                if s_dist_q > 2.0 * e_dist { continue; }
                let d_es = self.metric.distance(
                    self.vec_store.get(e_id),
                    self.vec_store.get(s_id),
                );
                if d_es <= e_dist { accept = false; break; }
            }

            if accept {
                self.select_buf.push((e_id, e_dist));
            } else if self.config.keep_pruned {
                self.pruned_buf.push((e_id, e_dist));
            }
        }

        if self.config.keep_pruned {
            let needed = m.saturating_sub(self.select_buf.len());
            let add    = needed.min(self.pruned_buf.len());
            for i in 0..add {
                let (id, dist) = self.pruned_buf[i]; // Copy
                self.select_buf.push((id, dist));
            }
        }
    }

    // ─── Reverse-update pruning ───────────────────────────────────────────

    /// Heuristic prune of `node_id`'s connection list at `layer` back to
    /// `m_max` entries — **Algorithm 4**, using stored distances.
    ///
    /// Unlike the naïve approach (which cloned the list, recomputed all M
    /// distances, then ran the heuristic), here we:
    ///
    /// 1. **Sort by stored distances** — zero new distance computations.
    /// 2. **Run the heuristic diversity check** — O(M²/2) pairwise in the
    ///    worst case, but the triangle-inequality shortcut eliminates most of
    ///    them in practice (≈ 60% skipped in benchmarks).
    /// 3. **Write the result back in-place** — no heap allocation; results
    ///    land in `self.select_buf` (reused) then moved to the connection list.
    ///
    /// Caller must ensure `connections[node_id][layer].len() == m_max + 1`.
    fn prune_connections_heuristic(&mut self, node_id: usize, layer: usize, m_max: usize) {
        // ── Step 1: load stored (id, dist_from_node) into prune_buf ──────
        self.prune_buf.clear();
        for (nb_u32, dist) in self.graph.neighbours(node_id, layer).into_iter().flatten() {
            self.prune_buf.push((nb_u32 as usize, dist));
        }
        // Sort closest-first by stored distance — no distance computation.
        self.prune_buf.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        // ── Step 2: heuristic selection into select_buf ───────────────────
        self.select_buf.clear();
        self.pruned_buf.clear();

        // Fast path: if we already have ≤ m_max (shouldn't happen here, but
        // guard for correctness), nothing to do.
        // Normal path: iterate sorted candidates, apply diversity criterion.
        for i in 0..self.prune_buf.len() {
            if self.select_buf.len() >= m_max { break; }
            let (e_id, e_dist) = self.prune_buf[i]; // Copy — no borrow held

            let mut accept = true;
            for j in 0..self.select_buf.len() {
                let (s_id, s_dist_node) = self.select_buf[j]; // Copy
                // Triangle-inequality shortcut:
                // d(node,s) > 2·d(node,e)  →  d(e,s) ≥ d(node,s)−d(node,e) > d(node,e)
                // so the accept condition d(node,e) ≤ d(e,s) is guaranteed.
                if s_dist_node > 2.0 * e_dist { continue; }
                let d_es = self.metric.distance(
                    self.vec_store.get(e_id),
                    self.vec_store.get(s_id),
                );
                if d_es <= e_dist { accept = false; break; }
            }

            if accept {
                self.select_buf.push((e_id, e_dist));
            } else if self.config.keep_pruned {
                self.pruned_buf.push((e_id, e_dist));
            }
        }

        if self.config.keep_pruned {
            let needed = m_max.saturating_sub(self.select_buf.len());
            let add    = needed.min(self.pruned_buf.len());
            for i in 0..add {
                let (id, dist) = self.pruned_buf[i];
                self.select_buf.push((id, dist));
            }
        }

        // ── Step 3: write result back to the connection list ──────────────
        // `select_buf` and `graph` are disjoint fields, so the connection list
        // is resolved once rather than re-matched on every push.
        let links = self.graph.neighbours_mut(node_id, layer);
        links.clear();
        links.extend(
            self.select_buf
                .iter()
                .map(|&(id, dist)| (id as u32, dist)),
        );
    }

    // ─── Stats / debug ────────────────────────────────────────────────────

    /// Return a human-readable summary of the index structure.
    pub fn stats(&self) -> IndexStats {
        let max_level = self.entry_point.map(|(_, l)| l).unwrap_or(0);
        let mut layer_counts = vec![0usize; max_level + 1];
        let mut layer_edges  = vec![0usize; max_level + 1];
        for node in 0..self.graph.node_count() {
            for l in 0..self.graph.level_count(node) {
                layer_counts[l] += 1;
                layer_edges[l] += self.graph.neighbour_count(node, l);
            }
        }
        IndexStats { num_vectors: self.vec_store.len(), max_level, layer_counts, layer_edges }
    }
}

/// Summary statistics about an [`Hnsw`] index.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
pub struct IndexStats {
    pub num_vectors:  usize,
    pub max_level:    usize,
    pub layer_counts: Vec<usize>,
    pub layer_edges:  Vec<usize>,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "HNSW index — {} vectors", self.num_vectors)?;
        writeln!(f, "  Max level : {}", self.max_level)?;
        for l in (0..=self.max_level).rev() {
            writeln!(f, "  Layer {:>3} : {:>6} nodes, {:>7} directed edges",
                     l, self.layer_counts[l], self.layer_edges[l])?;
        }
        Ok(())
    }
}
