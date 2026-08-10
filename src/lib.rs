//! # fast-hnsw
//!
//! A pure-Rust implementation of **Hierarchical Navigable Small World** (HNSW)
//! approximate nearest-neighbour search, following the algorithm from:
//!
//! > Malkov & Yashunin, *"Efficient and robust approximate nearest neighbor
//! > search using Hierarchical Navigable Small World graphs"*,
//! > IEEE TPAMI 2018.
//!
//! ## Quick start
//!
//! ```rust
//! use fast_hnsw::{Builder, Hnsw, SearchResult};
//! use fast_hnsw::distance::Euclidean;
//!
//! // Build an index.
//! let mut index: Hnsw<Euclidean> = Builder::new()
//!     .m(16)
//!     .ef_construction(200)
//!     .seed(42)
//!     .build(Euclidean).unwrap();
//!
//! // Insert vectors.
//! for i in 0..100_u32 {
//!     index.insert(vec![i as f32, (i * i) as f32]).unwrap();
//! }
//!
//! // Query: find 5 nearest neighbours with ef=50.
//! let results: Vec<SearchResult> = index.search(&[10.0, 101.0], 5, 50).unwrap();
//! assert_eq!(results[0].id, 10);
//! ```
//!
//! ## Distance metrics
//!
//! | Type                        | Description                         |
//! |-----------------------------|-------------------------------------|
//! | [`distance::Euclidean`]     | True L2 distance                    |
//! | [`distance::SquaredEuclidean`] | L2² (faster, same NN order)      |
//! | [`distance::Cosine`]        | 1 − cosine similarity               |
//! | [`distance::DotProduct`]    | 1 − dot product                     |
//! | [`distance::Manhattan`]     | L1 / taxicab distance               |
//!
//! Custom metrics are easy to add by implementing the [`distance::Distance`] trait.
//!
//! ## Feature flags
//!
//! Only `simd` is on by default, and it adds no dependencies: a default build
//! pulls in `rand` and `memmap2` and nothing else. Even with every feature
//! enabled the crate adds only `rayon`, and no build dependencies at all.
//!
//! | Feature | Adds | Effect |
//! |---------|------|--------|
//! | `simd` **(default)** | — | Hand-written NEON / AVX2 distance kernels; ~2× over the auto-vectorised fold. Disable for a `unsafe`-free portable build |
//! | `avx512` | — | 512-bit kernels, runtime-detected. ~10-15% over AVX2 at 512+ dims, parity below. **Raises the MSRV to 1.89** |
//! | `serde` | `serde` | `Serialize`/`Deserialize` for the crate's data types ([`Config`], [`SearchResult`], [`IndexStats`], ...). Indexes themselves are not serde types — use [`persist`] |
//! | `parallel` | `rayon` | `Builder::build_parallel` — multi-threaded construction, ~8× on 14 cores |
//! | `blas` | — | Routes [`DotProduct`](distance::DotProduct) and [`Cosine`](distance::Cosine) through a BLAS already on the system. **Measure first** — slower than the built-in SIMD on Apple silicon; see [`distance`] |
//!
//! ## Using a system BLAS
//!
//! The `blas` feature adds **no dependencies and no build dependencies**. This
//! crate never vendors, downloads, or compiles a BLAS; `build.rs` only emits a
//! link directive for one that already exists.
//!
//! | Target | Resolved as | Setup |
//! |--------|-------------|-------|
//! | macOS, iOS, tvOS, watchOS, visionOS | Accelerate framework | none — it ships in the SDK |
//! | Linux, *BSD | `pkg-config` → `openblas`, then `cblas` | `apt install libopenblas-dev` / `dnf install openblas-devel` |
//! | Windows | none by default | `vcpkg install openblas` or oneAPI MKL, then set the variables below |
//! | Anything else | falls back to `-lopenblas` with a build warning | set the variables below |
//!
//! Override the choice anywhere with environment variables at build time:
//!
//! ```sh
//! FAST_HNSW_BLAS_LIB=mkl_rt              # or: openblas, blis, "cblas,blas",
//!                                        #     or framework=Accelerate
//! FAST_HNSW_BLAS_LIB_DIR=/opt/blas/lib   # added to the link search path
//! FAST_HNSW_BLAS_STATIC=1                # link statically
//! ```
//!
//! iOS deserves a note: third-party native libraries cannot be installed
//! system-wide there, so Accelerate is effectively the only option — and
//! because it is part of the SDK, `features = ["blas"]` needs no configuration
//! on any Apple platform.
//!
//! Link an **LP64** build (32-bit indices). ILP64 builds such as MKL's ILP64
//! layer take 64-bit integers and are not ABI-compatible with the declarations
//! in this crate.
//!
//! The internal `simd` module dispatches to NEON on AArch64 and to AVX2+FMA on
//! x86-64 when the CPU reports them, falling back to a portable fold otherwise.
//!
//! ## Loading an index
//!
//! | Constructor | Vectors live | Graph lives | Notes |
//! |-------------|--------------|-------------|-------|
//! | [`persist::load`] | RAM | RAM | Simplest; needs room for everything |
//! | [`persist::load_mmap`] | Page cache | Page cache | Fastest for large indexes |
//! | [`PreadIndex::open`] | Disk | RAM | Bounded residency, I/O errors as values, NFS-safe — but slower; see [`pread`] |
//!
//! Both optional features involve trade-offs worth reading before enabling
//! them: the `parallel` module documents the recall cost of concurrent construction, and
//! [`distance`] documents why L2/L1 never use BLAS and why BLAS currently loses
//! to the built-in SIMD kernels here.
//!
//! Low-bit quantized sidecars live in the companion `fast-hnsw-quantized`
//! crate, whose `hnsw` feature (on by default) wires them into
//! [`Hnsw::search_with_distance`] so a graph can be traversed while vectors
//! are scored from a compressed mapping.

pub mod builder;
pub mod distance;
pub mod error;
pub(crate) mod heap;
pub mod hnsw;
pub mod labeled;
pub mod paired;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod payload;
#[cfg(feature = "blas")]
mod blas;
pub mod pread;
pub(crate) mod simd;
pub mod persist;

pub use builder::Builder;
pub use error::{Error, Result};
pub use hnsw::{
    Config, Hnsw, IndexStats, PruneStrategy, SearchResult, SearchWorkspace,
};
pub use labeled::{LabeledIndex, MappedLabeledIndex, MappedLabeledResult};
pub use paired::PairedIndex;
pub use pread::{PreadIndex, PreadVectors};

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use distance::{Cosine, Distance, Euclidean, Manhattan, SquaredEuclidean};
    use labeled::LabeledIndex;
    use paired::PairedIndex;
    use crate::persist;

    // ── helpers ──────────────────────────────────────────────────────────

    fn build_index(n: usize, dim: usize, seed: u64) -> Hnsw<Euclidean> {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed + 1_000);
        let mut index = Builder::new()
            .m(16)
            .ef_construction(200)
            .seed(seed)
            .build(Euclidean).unwrap();
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v).unwrap();
        }
        index
    }

    /// Brute-force exact k-NN.
    fn exact_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut dists: Vec<(f32, usize)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                (d, i)
            })
            .collect();
        dists.sort_by(|a, b| a.0.total_cmp(&b.0));
        dists.iter().take(k).map(|(_, i)| *i).collect()
    }

    // ── unit tests ────────────────────────────────────────────────────────

    #[test]
    fn empty_index_returns_nothing() {
        let index: Hnsw<Euclidean> = Builder::new().build(Euclidean).unwrap();
        assert!(index.search(&[1.0, 2.0], 5, 20).unwrap().is_empty());
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn single_vector_always_returned() {
        let mut index = Builder::new().seed(0).build(Euclidean).unwrap();
        index.insert(vec![1.0, 2.0, 3.0]).unwrap();
        let res = index.search(&[0.0, 0.0, 0.0], 1, 10).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn ids_are_assigned_sequentially() {
        let mut index = Builder::new().seed(1).build(Euclidean).unwrap();
        for i in 0..20 {
            let id = index.insert(vec![i as f32]).unwrap();
            assert_eq!(id, i);
        }
        assert_eq!(index.len(), 20);
    }

    #[test]
    fn nearest_of_two_is_correct() {
        let mut index = Builder::new().seed(2).build(Euclidean).unwrap();
        index.insert(vec![0.0, 0.0]).unwrap(); // id=0
        index.insert(vec![10.0, 0.0]).unwrap(); // id=1
        // Query very close to id=0
        let res = index.search(&[0.1, 0.0], 1, 10).unwrap();
        assert_eq!(res[0].id, 0);
        // Query very close to id=1
        let res = index.search(&[9.9, 0.0], 1, 10).unwrap();
        assert_eq!(res[0].id, 1);
    }

    #[test]
    fn distances_are_non_negative_and_ordered() {
        let index = build_index(200, 16, 3);
        let query: Vec<f32> = vec![0.5; 16];
        let results = index.search(&query, 10, 50).unwrap();
        assert_eq!(results.len(), 10);
        for w in results.windows(2) {
            assert!(w[0].distance >= 0.0);
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn k_larger_than_index_returns_all() {
        let index = build_index(30, 4, 4);
        let query = vec![0.5f32; 4];
        let res = index.search(&query, 100, 200).unwrap();
        assert_eq!(res.len(), 30);
    }

    #[test]
    fn filtered_search_applies_filter_before_top_k() {
        let mut index = Builder::new()
            .m(16)
            .ef_construction(100)
            .seed(44)
            .build(Euclidean).unwrap();
        for id in 0..100 {
            index.insert(vec![id as f32]).unwrap();
        }

        let results = index.search_filtered(&[12.1], 3, 100, |id| id % 10 == 0).unwrap();
        assert_eq!(
            results.iter().map(|result| result.id).collect::<Vec<_>>(),
            [10, 20, 0]
        );
    }

    #[test]
    fn filtered_search_navigates_through_rejected_nodes() {
        let mut index = Builder::new()
            .m(16)
            .ef_construction(100)
            .seed(45)
            .build(Euclidean).unwrap();
        for id in 0..100 {
            index.insert(vec![id as f32]).unwrap();
        }

        let results = index.search_filtered(&[0.0], 1, 100, |id| id == 99).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 99);
    }

    #[test]
    fn filtered_search_handles_empty_eligibility_set() {
        let index = build_index(100, 4, 46);
        let results = index.search_filtered(&[0.5; 4], 10, 100, |_| false).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn reusable_workspace_preserves_search_and_filter_results() {
        let index = build_index(200, 16, 47);
        let query = [0.5; 16];
        let expected = index.search(&query, 10, 50).unwrap();
        let expected_filtered = index.search_filtered(&query, 10, 50, |id| id % 3 == 0).unwrap();

        let mut workspace = SearchWorkspace::default();
        assert_eq!(
            index.search_with_workspace(&query, 10, 50, &mut workspace).unwrap(),
            expected
        );
        assert_eq!(
            index.search_filtered_with_workspace(
                &query,
                10,
                50,
                |id| id % 3 == 0,
                &mut workspace,
            ).unwrap(),
            expected_filtered
        );
        assert_eq!(
            index.search_with_workspace(&query, 10, 100, &mut workspace).unwrap(),
            index.search(&query, 10, 100).unwrap()
        );

        let transformed = index
            .try_search_with_workspace_map(
                &query,
                10,
                50,
                &mut workspace,
                |result| Ok::<_, Error>((result.id, result.distance.to_bits())),
            )
            .expect("result transform failed");
        assert_eq!(
            transformed,
            expected
                .iter()
                .map(|result| (result.id, result.distance.to_bits()))
                .collect::<Vec<_>>()
        );

        let transformed_filtered = index
            .try_search_filtered_with_workspace_map(
                &query,
                10,
                50,
                |id| id % 3 == 0,
                &mut workspace,
                |result| Ok::<_, Error>((result.id, result.distance.to_bits())),
            )
            .expect("filtered result transform failed");
        assert_eq!(
            transformed_filtered,
            expected_filtered
                .iter()
                .map(|result| (result.id, result.distance.to_bits()))
                .collect::<Vec<_>>()
        );

        let transform_error = index.try_search_with_workspace_map(
            &query,
            10,
            50,
            &mut workspace,
            |_| Err::<(), _>(std::io::Error::other("transform failed")),
        );
        assert_eq!(
            transform_error.expect_err("transform should fail").to_string(),
            "transform failed"
        );
    }

    #[test]
    fn external_distance_search_preserves_results_and_filtering() {
        let index = build_index(200, 16, 48);
        let vectors = (0..index.len())
            .map(|id| index.get_vector(id).unwrap().to_vec())
            .collect::<Vec<_>>();
        let query = [0.35; 16];
        // Delegate to the index's own metric.  The assertions below compare
        // distances bit-for-bit, and the built-in kernels accumulate across
        // several independent lanes; an open-coded serial fold here would
        // differ from them in the last ULP even though it computes the same
        // quantity.  What this test is about is that external-distance
        // traversal reproduces internal scoring, not float associativity.
        let distance = |id: usize| Euclidean.distance(&vectors[id], &query);

        assert_eq!(
            index.search_with_distance(10, 75, distance).unwrap(),
            index.search(&query, 10, 75).unwrap()
        );
        assert_eq!(
            index.search_filtered_with_distance(10, 75, distance, |id| id % 4 == 0).unwrap(),
            index.search_filtered(&query, 10, 75, |id| id % 4 == 0).unwrap()
        );

        let mut workspace = SearchWorkspace::default();
        assert_eq!(
            index.search_with_distance_and_workspace(10, 75, distance, &mut workspace).unwrap(),
            index.search(&query, 10, 75).unwrap()
        );
        assert_eq!(
            index.search_filtered_with_distance_and_workspace(
                10,
                75,
                distance,
                |id| id % 4 == 0,
                &mut workspace,
            ).unwrap(),
            index.search_filtered(&query, 10, 75, |id| id % 4 == 0).unwrap()
        );

        let transformed = index
            .try_search_with_distance_and_workspace_map(
                10,
                75,
                distance,
                &mut workspace,
                |result| Ok::<_, Error>((result.id, result.distance.to_bits())),
            )
            .expect("external-distance transform failed");
        assert_eq!(
            transformed,
            index
                .search(&query, 10, 75)
                .unwrap()
                .iter()
                .map(|result| (result.id, result.distance.to_bits()))
                .collect::<Vec<_>>()
        );

        let transformed_filtered = index
            .try_search_filtered_with_distance_and_workspace_map(
                10,
                75,
                distance,
                |id| id % 4 == 0,
                &mut workspace,
                |result| Ok::<_, Error>((result.id, result.distance.to_bits())),
            )
            .expect("filtered external-distance transform failed");
        assert_eq!(
            transformed_filtered,
            index
                .search_filtered(&query, 10, 75, |id| id % 4 == 0)
                .unwrap()
                .iter()
                .map(|result| (result.id, result.distance.to_bits()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn stored_vectors_are_retrievable() {
        let mut index = Builder::new().seed(5).build(Euclidean).unwrap();
        let vecs = vec![vec![1.0f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        for v in &vecs {
            index.insert(v.clone()).unwrap();
        }
        for (i, v) in vecs.iter().enumerate() {
            assert_eq!(index.get_vector(i).unwrap(), v.as_slice());
        }
    }

    #[test]
    fn dim_is_tracked() {
        let mut index = Builder::new().seed(6).build(Euclidean).unwrap();
        assert_eq!(index.dim(), None);
        index.insert(vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(index.dim(), Some(3));
    }

    #[test]
    fn wrong_dimension_is_reported() {
        let mut index = Builder::new().seed(7).build(Euclidean).unwrap();
        index.insert(vec![1.0, 2.0, 3.0]).unwrap();
        match index.insert(vec![1.0, 2.0]) {
            Err(Error::DimensionMismatch { expected: 3, actual: 2 }) => {}
            other => panic!("expected a dimension mismatch, got {other:?}"),
        }
    }

    // ── Query-dimension validation ────────────────────────────────────────
    //
    // Every metric folds with `zip` semantics, so an unchecked query of the
    // wrong length silently scores against a truncated vector and returns
    // confident, wrong neighbours instead of failing.

    fn two_dimensional_index() -> Hnsw<Euclidean> {
        let mut index = Builder::new().seed(70).build(Euclidean).unwrap();
        for id in 0..50 {
            index.insert(vec![0.0, id as f32]).unwrap();
        }
        index
    }

    #[test]
    fn short_query_is_rejected() {
        match two_dimensional_index().search(&[0.0], 1, 50) {
            Err(Error::DimensionMismatch { expected: 2, .. }) => {}
            other => panic!("expected a dimension mismatch, got {other:?}"),
        }
    }

    #[test]
    fn long_query_is_rejected() {
        match two_dimensional_index().search(&[0.0, 30.0, 1.0], 1, 50) {
            Err(Error::DimensionMismatch { expected: 2, .. }) => {}
            other => panic!("expected a dimension mismatch, got {other:?}"),
        }
    }

    #[test]
    fn filtered_search_rejects_mismatched_query() {
        match two_dimensional_index().search_filtered(&[0.0], 1, 50, |_| true) {
            Err(Error::DimensionMismatch { expected: 2, .. }) => {}
            other => panic!("expected a dimension mismatch, got {other:?}"),
        }
    }

    #[test]
    fn workspace_search_rejects_mismatched_query() {
        let mut workspace = SearchWorkspace::default();
        match two_dimensional_index().search_with_workspace(&[0.0], 1, 50, &mut workspace) {
            Err(Error::DimensionMismatch { expected: 2, actual: 1 }) => {}
            other => panic!("expected a dimension mismatch, got {other:?}"),
        }
    }

    #[test]
    fn zero_k_is_reported() {
        let index = two_dimensional_index();
        assert!(matches!(index.search(&[0.0, 1.0], 0, 10), Err(Error::ZeroK)));
        assert!(matches!(index.exact_search(&[0.0, 1.0], 0), Err(Error::ZeroK)));
    }

    #[test]
    fn matching_query_dimension_is_accepted() {
        let index = two_dimensional_index();
        assert_eq!(index.search(&[0.0, 30.0], 1, 50).unwrap()[0].id, 30);
    }

    #[test]
    fn empty_index_accepts_any_query_dimension() {
        // No dimension has been established yet, and the search returns
        // nothing regardless, so there is nothing to validate against.
        let index: Hnsw<Euclidean> = Builder::new().build(Euclidean).unwrap();
        assert!(index.search(&[1.0, 2.0, 3.0], 5, 20).unwrap().is_empty());
    }

    // ── Mapped vector bounds ──────────────────────────────────────────────

    #[test]
    fn mmap_get_vector_rejects_out_of_range_id() {
        // `get_vector` is safe and public; on a mapped index it used to index
        // raw pointers without a check, so an out-of-range id was undefined
        // behaviour. It is now a reported error.
        let (index, _) = make_hnsw(20, 8, 311);
        let dir = tempdir();
        let path = dir.path().join("mmap-bounds.hnsw");
        persist::save(&index, &path).expect("save failed");
        let mapped = persist::load_mmap(&path, Euclidean).expect("mmap load failed");
        assert_eq!(mapped.get_vector(19).unwrap().len(), 8);
        assert!(matches!(
            mapped.get_vector(20),
            Err(Error::IdOutOfBounds { id: 20, len: 20 })
        ));
    }

    // ── Owned-load validation parity with the mapped path ─────────────────

    /// Overwrite `len` bytes at `offset` in `path`.
    fn patch(path: &std::path::Path, offset: u64, bytes: &[u8]) {
        use std::io::{Seek, SeekFrom, Write};
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open failed");
        file.seek(SeekFrom::Start(offset)).expect("seek failed");
        file.write_all(bytes).expect("write failed");
    }

    #[test]
    fn owned_load_rejects_out_of_range_neighbour() {
        use std::io::{Read, Seek, SeekFrom};

        let (n, dim) = (20usize, 8usize);
        let (index, _) = make_hnsw(n, dim, 312);
        let dir = tempdir();
        let path = dir.path().join("owned-invalid-neighbour.hnsw");
        persist::save(&index, &path).expect("save failed");

        // Point the first neighbour of node 0 at a node that does not exist.
        let offsets_start = 256 + n * dim * 4 + n * 4;
        let mut file = std::fs::File::open(&path).expect("open failed");
        file.seek(SeekFrom::Start(offsets_start as u64)).expect("seek failed");
        let mut offset = [0u8; 8];
        file.read_exact(&mut offset).expect("offset read failed");
        let first_record = u64::from_le_bytes(offset);
        drop(file);
        patch(&path, first_record + 4, &(n as u32).to_le_bytes());

        let error = persist::load(&path, Euclidean)
            .err()
            .expect("out-of-range neighbour should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("invalid neighbour id"));
    }

    #[test]
    fn owned_load_rejects_invalid_connection_offset() {
        let (n, dim) = (20usize, 8usize);
        let (index, _) = make_hnsw(n, dim, 313);
        let dir = tempdir();
        let path = dir.path().join("owned-invalid-offset.hnsw");
        persist::save(&index, &path).expect("save failed");

        let offsets_start = 256 + n * dim * 4 + n * 4;
        patch(&path, offsets_start as u64, &0u64.to_le_bytes());

        let error = persist::load(&path, Euclidean)
            .err()
            .expect("invalid connection offset should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("invalid connection offset"));
    }

    #[test]
    fn owned_load_rejects_implausible_node_level() {
        let (n, dim) = (20usize, 8usize);
        let (index, _) = make_hnsw(n, dim, 314);
        let dir = tempdir();
        let path = dir.path().join("owned-invalid-level.hnsw");
        persist::save(&index, &path).expect("save failed");

        let levels_start = 256 + n * dim * 4;
        patch(&path, levels_start as u64, &u32::MAX.to_le_bytes());

        let error = persist::load(&path, Euclidean)
            .err()
            .expect("implausible node level should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("supported maximum"));
    }

    #[test]
    fn owned_load_rejects_vector_section_larger_than_file() {
        let (index, _) = make_hnsw(20, 8, 315);
        let dir = tempdir();
        let path = dir.path().join("owned-vector-overflow.hnsw");
        persist::save(&index, &path).expect("save failed");

        // A huge `n` used to reach `vec![0u8; n * dim * 4]` before anything
        // checked it against the actual file length.
        patch(&path, 12, &(u32::MAX as u64).to_le_bytes());

        let error = persist::load(&path, Euclidean)
            .err()
            .expect("oversized vector section should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("file too short"));
    }

    #[test]
    fn owned_load_rejects_corrupt_payload_offset_table() {
        let mut index: LabeledIndex<Euclidean, String> =
            Builder::new().seed(316).build_labeled(Euclidean).unwrap();
        for id in 0..8u32 {
            index.insert(vec![id as f32, 1.0], format!("item-{id}")).unwrap();
        }
        let dir = tempdir();
        let path = dir.path().join("owned-corrupt-payload-offsets.hnsw");
        index.save(&path).expect("save failed");

        // Find the variable-width offset table: it follows the 16-byte payload
        // header, which itself follows the graph.  Locating it exactly is
        // fiddly, so instead corrupt every 8-byte-aligned slot in the tail and
        // require that the loader always reports an error rather than
        // attempting a colossal allocation.
        let bytes = std::fs::read(&path).expect("read failed");
        let tail_start = bytes.len().saturating_sub(160);
        let mut rejected = 0;
        for slot in (tail_start..bytes.len().saturating_sub(8)).step_by(8) {
            std::fs::write(&path, &bytes).expect("restore failed");
            patch(&path, slot as u64, &u64::MAX.to_le_bytes());
            if LabeledIndex::<Euclidean, String>::load(&path, Euclidean).is_err() {
                rejected += 1;
            }
        }
        assert!(rejected > 0, "expected at least one corrupt offset to be rejected");
    }

    // ── deletion ──────────────────────────────────────────────────────────

    #[test]
    fn removed_vectors_disappear_from_results() {
        let mut index = Builder::new().seed(600).build(Euclidean).unwrap();
        for id in 0..100 {
            index.insert(vec![id as f32]).unwrap();
        }
        assert_eq!(index.search(&[50.0], 1, 50).unwrap()[0].id, 50);

        assert!(index.remove(50));
        assert!(index.is_deleted(50));
        assert_eq!(index.deleted_count(), 1);
        assert_eq!(index.live_len(), 99);
        // `len` still counts the slot, so surviving ids do not shift.
        assert_eq!(index.len(), 100);

        let hits = index.search(&[50.0], 5, 50).unwrap();
        assert!(hits.iter().all(|hit| hit.id != 50));
        // The nearest survivors are the immediate neighbours.
        assert!(hits.iter().any(|hit| hit.id == 49));
        assert!(hits.iter().any(|hit| hit.id == 51));
    }

    #[test]
    fn remove_is_idempotent_and_range_checked() {
        let mut index = Builder::new().seed(601).build(Euclidean).unwrap();
        index.insert(vec![1.0]).unwrap();
        assert!(index.remove(0));
        assert!(!index.remove(0), "second removal should report no change");
        assert!(!index.remove(99), "out-of-range id should report no change");
        assert_eq!(index.deleted_count(), 1);
        assert!(index.search(&[1.0], 1, 10).unwrap().is_empty());
    }

    #[test]
    fn restore_makes_a_vector_visible_again() {
        let mut index = Builder::new().seed(602).build(Euclidean).unwrap();
        for id in 0..20 {
            index.insert(vec![id as f32]).unwrap();
        }
        assert!(index.remove(7));
        assert!(index.search(&[7.0], 3, 20).unwrap().iter().all(|hit| hit.id != 7));
        assert!(index.restore(7));
        assert!(!index.restore(7), "second restore should report no change");
        assert_eq!(index.deleted_count(), 0);
        assert_eq!(index.search(&[7.0], 1, 20).unwrap()[0].id, 7);
    }

    /// Deleted nodes must keep relaying traversal. If tombstoning severed the
    /// graph, the survivors behind a wall of deleted nodes would become
    /// unreachable rather than merely unranked.
    #[test]
    fn deleted_nodes_still_relay_traversal() {
        let mut index = Builder::new()
            .m(16)
            .ef_construction(200)
            .seed(603)
            .build(Euclidean).unwrap();
        for id in 0..500 {
            index.insert(vec![id as f32]).unwrap();
        }
        // Delete a wide contiguous band between the query and the survivors.
        for id in 100..450 {
            index.remove(id);
        }
        assert_eq!(index.deleted_count(), 350);
        assert_eq!(index.live_len(), 150);

        let hits = index.search(&[300.0], 5, 200).unwrap();
        assert_eq!(hits.len(), 5);
        assert!(
            hits.iter().all(|hit| !(100..450).contains(&hit.id)),
            "deleted band leaked into results: {hits:?}"
        );
        // The closest survivors on either side of the band are reachable.
        let ids: Vec<usize> = hits.iter().map(|hit| hit.id).collect();
        assert!(ids.contains(&99) || ids.contains(&450), "got {ids:?}");
    }

    #[test]
    fn deletion_composes_with_a_user_filter() {
        let mut index = Builder::new().seed(604).build(Euclidean).unwrap();
        for id in 0..100 {
            index.insert(vec![id as f32]).unwrap();
        }
        index.remove(20);
        index.remove(40);
        let hits = index.search_filtered(&[0.0], 5, 100, |id| id % 20 == 0).unwrap();
        let ids: Vec<usize> = hits.iter().map(|hit| hit.id).collect();
        // Multiples of 20, minus the two tombstoned ones.
        assert_eq!(ids, [0, 60, 80]);
    }

    #[test]
    fn deletion_applies_to_external_distance_search() {
        let index_source = build_index(200, 8, 605);
        let vectors: Vec<Vec<f32>> = (0..index_source.len())
            .map(|id| index_source.get_vector(id).unwrap().to_vec())
            .collect();
        let mut index = index_source;
        let query = [0.5f32; 8];
        let target = index.search(&query, 1, 50).unwrap()[0].id;
        assert!(index.remove(target));

        let distance = |id: usize| Euclidean.distance(&vectors[id], &query);
        let hits = index.search_with_distance(5, 50, distance).unwrap();
        assert!(
            hits.iter().all(|hit| hit.id != target),
            "external-distance search ignored the tombstone"
        );
    }

    /// The entry point is where every search starts descending. Tombstoning it
    /// must hide it from results without stranding the layers below it.
    #[test]
    fn deleting_the_entry_point_keeps_the_index_searchable() {
        let mut index = build_index(300, 8, 606);
        let (entry_id, entry_level) = index.entry_point.expect("index should have an entry point");
        assert!(entry_level >= 1, "expected a multi-layer graph to make this meaningful");

        assert!(index.remove(entry_id));
        // The entry point is unchanged: it still routes traversal, it just
        // cannot be returned any more.
        assert_eq!(index.entry_point, Some((entry_id, entry_level)));

        let hits = index.search(&[0.5; 8], 5, 100).unwrap();
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|hit| hit.id != entry_id));
        assert!(hits.iter().all(|hit| !index.is_deleted(hit.id)));
    }

    #[test]
    fn compacted_rebuilds_without_deleted_vectors() {
        let mut index = Builder::new().m(8).ef_construction(64).seed(607).build(Euclidean).unwrap();
        for id in 0..100 {
            index.insert(vec![id as f32]).unwrap();
        }
        for id in (0..100).step_by(2) {
            index.remove(id);
        }
        assert_eq!(index.live_len(), 50);

        let (compacted, old_ids) = index.compacted(Euclidean, Some(607)).unwrap();
        assert_eq!(compacted.len(), 50);
        assert_eq!(compacted.live_len(), 50);
        assert_eq!(compacted.deleted_count(), 0);
        assert_eq!(old_ids.len(), 50);
        // New id n corresponds to old id 2n+1, and holds the same vector.
        for (new_id, &old_id) in old_ids.iter().enumerate() {
            assert_eq!(old_id, new_id * 2 + 1);
            assert_eq!(compacted.get_vector(new_id).unwrap(), index.get_vector(old_id).unwrap());
        }
        // Only odd values survive, so 49.0 and 51.0 are tied at distance 1.0
        // from the query and either is a correct answer.
        let hit = &compacted.search(&[50.0], 1, 50).unwrap()[0];
        let value = compacted.get_vector(hit.id).unwrap()[0];
        assert!(value == 49.0 || value == 51.0, "got {value}");
        assert_eq!(hit.distance, 1.0);
    }

    #[test]
    fn saving_a_tombstoned_index_is_refused() {
        let mut index = Builder::new().seed(608).build(Euclidean).unwrap();
        for id in 0..20 {
            index.insert(vec![id as f32]).unwrap();
        }
        index.remove(3);
        let dir = tempdir();
        let path = dir.path().join("tombstoned.hnsw");

        // Silently writing would resurrect id 3 on the next load.
        let error = persist::save(&index, &path)
            .expect_err("saving a tombstoned index should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("compacted"), "{error}");
        assert!(persist::save_compact(&index, &path).is_err());

        // Compacting first makes it saveable, and the deleted vector is gone.
        let (compacted, _) = index.compacted(Euclidean, Some(608)).unwrap();
        persist::save(&compacted, &path).expect("compacted save failed");
        let loaded = persist::load(&path, Euclidean).expect("load failed");
        assert_eq!(loaded.len(), 19);
        assert!((0..loaded.len()).all(|id| loaded.get_vector(id).unwrap() != [3.0]));
    }

    #[test]
    fn labeled_deletion_keeps_payloads_aligned() {
        let mut index: LabeledIndex<Euclidean, String> =
            Builder::new().seed(609).build_labeled(Euclidean).unwrap();
        for id in 0..20u32 {
            index.insert(vec![id as f32], format!("item-{id}")).unwrap();
        }
        assert!(index.remove(5));
        assert_eq!(index.live_len(), 19);
        assert!(index.search(&[5.0], 3, 20).unwrap().iter().all(|hit| hit.id != 5));
        // Surviving ids keep addressing their original payloads.
        assert_eq!(index.get_payload(6).unwrap(), "item-6");

        let compacted = index.compacted(Euclidean, Some(609)).unwrap();
        assert_eq!(compacted.len(), 19);
        assert!(
            (0..compacted.len()).all(|id| compacted.get_payload(id).unwrap() != "item-5"),
            "deleted payload survived compaction"
        );
        // Renumbered, but vector and payload still correspond.
        for id in 0..compacted.len() {
            let expected = format!("item-{}", compacted.get_embedding(id).unwrap()[0] as u32);
            assert_eq!(compacted.get_payload(id).unwrap(), &expected);
        }
    }

    #[test]
    fn deletion_works_on_a_memory_mapped_index() {
        let (index, _) = make_hnsw(100, 8, 610);
        let dir = tempdir();
        let path = dir.path().join("mmap-delete.hnsw");
        persist::save(&index, &path).expect("save failed");
        let mut mapped = persist::load_mmap(&path, Euclidean).expect("mmap load failed");

        let target = mapped.search(&[0.5; 8], 1, 50).unwrap()[0].id;
        // Tombstones live in memory, so a read-only mapping can still be
        // filtered without writing to the file.
        assert!(mapped.remove(target));
        assert_eq!(mapped.live_len(), 99);
        assert!(mapped
            .search(&[0.5; 8], 5, 50).unwrap()
            .iter()
            .all(|hit| hit.id != target));
    }

    #[test]
    fn no_deletions_means_no_tombstone_allocation_cost() {
        // The fast unfiltered path must remain in use when nothing is deleted.
        let index = build_index(100, 4, 611);
        assert_eq!(index.deleted_count(), 0);
        assert_eq!(index.live_len(), index.len());
        assert!(!index.is_deleted(0));
        assert!(!index.is_deleted(usize::MAX));
    }

    // ── fallible insert ───────────────────────────────────────────────────

    #[test]
    fn insert_reports_dimension_mismatch() {
        let mut index = Builder::new().seed(620).build(Euclidean).unwrap();
        assert_eq!(index.insert(vec![1.0, 2.0, 3.0]).unwrap(), 0);
        match index.insert(vec![1.0, 2.0]) {
            Err(Error::DimensionMismatch { expected: 3, actual: 2 }) => {}
            other => panic!("expected a dimension mismatch, got {other:?}"),
        }
        // The rejected insert must not have consumed an id or corrupted state.
        assert_eq!(index.len(), 1);
        assert_eq!(index.insert(vec![4.0, 5.0, 6.0]).unwrap(), 1);
    }

    #[test]
    fn insert_reports_read_only_mapping() {
        let (index, _) = make_hnsw(20, 8, 621);
        let dir = tempdir();
        let path = dir.path().join("readonly.hnsw");
        persist::save(&index, &path).expect("save failed");
        let mut mapped = persist::load_mmap(&path, Euclidean).expect("mmap load failed");
        match mapped.insert(vec![0.0; 8]) {
            Err(Error::ReadOnly) => {}
            other => panic!("expected ReadOnly, got {other:?}"),
        }
    }

    // ── exact search ──────────────────────────────────────────────────────

    /// Brute force is the reference the approximate search is measured
    /// against, so it must agree with an independent scalar computation —
    /// including on the BLAS `sgemv` path, which computes the whole score
    /// vector in one call.
    #[test]
    fn exact_search_matches_a_scalar_scan() {
        use distance::DotProduct;

        let dim = 24;
        let mut state = 12345u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let vectors: Vec<Vec<f32>> =
            (0..300).map(|_| (0..dim).map(|_| next()).collect()).collect();

        let mut index: Hnsw<DotProduct> =
            Builder::new().seed(870).build(DotProduct).unwrap();
        for v in &vectors {
            index.insert(v.clone()).unwrap();
        }
        let query: Vec<f32> = (0..dim).map(|_| next()).collect();

        let mut reference: Vec<(f32, usize)> = vectors
            .iter()
            .enumerate()
            .map(|(id, v)| {
                let dot: f32 = v.iter().zip(&query).map(|(a, b)| a * b).sum();
                (1.0 - dot, id)
            })
            .collect();
        reference.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

        let got = index.exact_search(&query, 10).unwrap();
        assert_eq!(got.len(), 10);
        for (rank, hit) in got.iter().enumerate() {
            assert_eq!(hit.id, reference[rank].1, "rank {rank} differs");
            assert!((hit.distance - reference[rank].0).abs() <= 1e-4);
        }
        assert!(got.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    #[test]
    fn exact_search_is_at_least_as_good_as_the_graph() {
        let index = build_index(400, 16, 871);
        let query = vec![0.5f32; 16];
        let exact = index.exact_search(&query, 10).unwrap();
        let approx = index.search(&query, 10, 200).unwrap();
        assert_eq!(exact.len(), 10);
        for (e, a) in exact.iter().zip(&approx) {
            assert!(e.distance <= a.distance + 1e-6);
        }
    }

    #[test]
    fn exact_search_respects_tombstones_and_edge_cases() {
        let mut index = build_index(120, 8, 872);
        let query = [0.5f32; 8];
        let target = index.exact_search(&query, 1).unwrap()[0].id;
        assert!(index.remove(target));
        assert!(index
            .exact_search(&query, 5)
            .unwrap()
            .iter()
            .all(|h| h.id != target));
        assert_eq!(index.exact_search(&query, 1_000).unwrap().len(), 119);

        let empty: Hnsw<Euclidean> = Builder::new().build(Euclidean).unwrap();
        assert!(empty.exact_search(&[0.0, 1.0], 5).unwrap().is_empty());
    }

    #[test]
    fn exact_search_validates_query_dimension() {
        let index = build_index(20, 8, 873);
        assert!(matches!(
            index.exact_search(&[0.5; 3], 1),
            Err(Error::DimensionMismatch { expected: 8, actual: 3 })
        ));
    }

    // ── serde ─────────────────────────────────────────────────────────────

    #[cfg(feature = "serde")]
    #[test]
    fn data_types_round_trip_through_serde() {
        let config = Config {
            m: 24,
            m0: Some(48),
            ef_construction: 321,
            use_heuristic: false,
            extend_candidates: true,
            keep_pruned: false,
            prune_strategy: PruneStrategy::Heuristic,
            capacity: 99,
        };
        let json = serde_json::to_string(&config).expect("serialize Config");
        let back: Config = serde_json::from_str(&json).expect("deserialize Config");
        assert_eq!(back.m, config.m);
        assert_eq!(back.prune_strategy, config.prune_strategy);

        // A round-tripped config must still build a working index.
        let mut index = Hnsw::new(back, Euclidean).unwrap();
        index.insert(vec![1.0, 2.0]).unwrap();
        assert_eq!(index.search(&[1.0, 2.0], 1, 10).unwrap()[0].id, 0);

        let results = build_index(50, 4, 880).search(&[0.5; 4], 3, 20).unwrap();
        let json = serde_json::to_string(&results).expect("serialize results");
        let back: Vec<SearchResult> =
            serde_json::from_str(&json).expect("deserialize results");
        assert_eq!(back, results);

        let stats = build_index(50, 4, 881).stats();
        let json = serde_json::to_string(&stats).expect("serialize stats");
        let back: IndexStats = serde_json::from_str(&json).expect("deserialize stats");
        assert_eq!(back.num_vectors, stats.num_vectors);
        assert_eq!(back.layer_counts, stats.layer_counts);
    }

    // ── PairedIndex deletion ──────────────────────────────────────────────
    //
    // Both graphs share one id space, so a deletion that reached only one side
    // would leave `search_by_a` and `search_by_b` disagreeing about which
    // items exist. Every test here checks both sides.

    fn paired_fixture() -> PairedIndex<Euclidean, Euclidean> {
        let mut index: PairedIndex<Euclidean, Euclidean> = Builder::new()
            .m(16)
            .ef_construction(100)
            .seed(850)
            .build_paired(Euclidean, Euclidean)
            .unwrap();
        for id in 0..40u32 {
            index
                .insert(vec![id as f32, 0.0], vec![0.0, id as f32])
                .unwrap();
        }
        index
    }

    #[test]
    fn paired_remove_hides_the_item_from_both_sides() {
        let mut index = paired_fixture();
        assert_eq!(index.search_by_a(&[20.0, 0.0], 1, 40).unwrap()[0].id, 20);
        assert_eq!(index.search_by_b(&[0.0, 20.0], 1, 40).unwrap()[0].id, 20);

        assert!(index.remove(20));
        assert!(index.is_deleted(20));
        assert_eq!(index.deleted_count(), 1);
        assert_eq!(index.live_len(), 39);
        assert_eq!(index.len(), 40, "ids must not shift");

        assert!(index
            .search_by_a(&[20.0, 0.0], 5, 40)
            .unwrap()
            .iter()
            .all(|h| h.id != 20));
        assert!(index
            .search_by_b(&[0.0, 20.0], 5, 40)
            .unwrap()
            .iter()
            .all(|h| h.id != 20));
        assert_eq!(index.index_a.deleted_count(), index.index_b.deleted_count());
        assert!(index.index_a.is_deleted(20) && index.index_b.is_deleted(20));
    }

    #[test]
    fn paired_remove_is_idempotent_and_restore_works() {
        let mut index = paired_fixture();
        assert!(index.remove(5));
        assert!(!index.remove(5));
        assert!(!index.remove(999));
        assert_eq!(index.deleted_count(), 1);

        assert!(index.restore(5));
        assert!(!index.restore(5));
        assert_eq!(index.deleted_count(), 0);
        assert_eq!(index.search_by_a(&[5.0, 0.0], 1, 40).unwrap()[0].id, 5);
        assert_eq!(index.search_by_b(&[0.0, 5.0], 1, 40).unwrap()[0].id, 5);
    }

    #[test]
    fn paired_compaction_renumbers_both_sides_identically() {
        let mut index = paired_fixture();
        for id in (0..40).step_by(2) {
            index.remove(id);
        }
        assert_eq!(index.live_len(), 20);

        let compacted = index
            .compacted(Euclidean, Euclidean, Some(850))
            .unwrap();
        assert_eq!(compacted.len(), 20);
        assert_eq!(compacted.deleted_count(), 0);

        // The pairing is the invariant: A and B must still describe the same
        // original item (odd ids) for every surviving slot.
        for id in 0..compacted.len() {
            let a = compacted.get_emb_a(id).unwrap();
            let b = compacted.get_emb_b(id).unwrap();
            assert_eq!(a[0], b[1], "A/B embeddings desynchronised at id {id}");
            assert_eq!(a[0] as usize % 2, 1, "an even (deleted) id survived");
        }
        let hits = compacted.search_by_a(&[21.0, 0.0], 1, 40).unwrap();
        assert_eq!(hits[0].emb_a[0], 21.0);
        assert_eq!(hits[0].emb_b[1], 21.0);
    }

    #[test]
    fn paired_save_rejects_tombstones_with_pairing_aware_advice() {
        let mut index = paired_fixture();
        index.remove(3);
        let dir = tempdir();
        let base = dir.path().join("paired-tombstoned");

        let error = index
            .save(&base)
            .expect_err("saving a tombstoned paired index must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("PairedIndex::compacted"), "{error}");
        assert!(
            !std::path::Path::new(&format!("{}_a.hnsw", base.display())).exists(),
            "A side was written before the pairing check ran"
        );

        let compacted = index.compacted(Euclidean, Euclidean, Some(850)).unwrap();
        compacted.save(&base).expect("compacted save failed");
        let loaded =
            PairedIndex::<Euclidean, Euclidean>::load(&base, Euclidean, Euclidean)
                .expect("load failed");
        assert_eq!(loaded.len(), 39);
        assert!((0..loaded.len()).all(|id| loaded.get_emb_a(id).unwrap()[0] != 3.0));
    }

    #[test]
    fn paired_compaction_rejects_desynchronised_sides() {
        // Deleting through the public fields bypasses the paired API; the two
        // sides would renumber differently, so compaction must refuse.
        let mut index = paired_fixture();
        index.index_a.remove(4);
        assert!(matches!(
            index.compacted(Euclidean, Euclidean, Some(850)),
            Err(Error::DesynchronizedPair)
        ));
    }

    // ── metric identity ───────────────────────────────────────────────────
    //
    // A graph's edges encode the metric that built it, so reopening a snapshot
    // with a different metric used to return confident, wrong neighbours.

    fn metric_sensitive_index() -> (Hnsw<Cosine>, [f32; 2]) {
        let mut index: Hnsw<Cosine> = Builder::new().seed(800).build(Cosine).unwrap();
        index.insert(vec![10.0, 0.0]).unwrap(); // id 0 — same direction, far in L2
        index.insert(vec![0.9, 0.9]).unwrap(); // id 1 — 45° off, near in L2
        (index, [1.0, 0.0])
    }

    #[test]
    fn wrong_metric_would_have_changed_the_answer() {
        let (index, query) = metric_sensitive_index();
        assert_eq!(index.search(&query, 1, 10).unwrap()[0].id, 0);
        let mut euclid: Hnsw<Euclidean> =
            Builder::new().seed(800).build(Euclidean).unwrap();
        euclid.insert(vec![10.0, 0.0]).unwrap();
        euclid.insert(vec![0.9, 0.9]).unwrap();
        assert_eq!(euclid.search(&query, 1, 10).unwrap()[0].id, 1);
    }

    #[test]
    fn loading_with_a_different_metric_is_rejected() {
        let (index, _) = metric_sensitive_index();
        let dir = tempdir();
        let path = dir.path().join("metric.hnsw");
        persist::save(&index, &path).expect("save failed");

        persist::load(&path, Cosine).expect("same-metric load should succeed");

        let error = persist::load(&path, Euclidean)
            .err()
            .expect("cross-metric load must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        let message = error.to_string();
        assert!(message.contains("metric mismatch"), "{message}");
        assert!(message.contains("Cosine"), "{message}");
        assert!(message.contains("Euclidean"), "{message}");
    }

    #[test]
    fn every_load_path_checks_the_metric() {
        use crate::pread::PreadIndex;

        let (index, _) = metric_sensitive_index();
        let dir = tempdir();
        let path = dir.path().join("metric-all-paths.hnsw");
        persist::save(&index, &path).expect("save failed");
        let compact = dir.path().join("metric-compact.hnsw");
        persist::save_compact(&index, &compact).expect("compact save failed");

        assert!(persist::load(&path, Euclidean).is_err(), "load");
        assert!(persist::load_mmap(&path, Euclidean).is_err(), "load_mmap");
        assert!(persist::load_mmap(&compact, Euclidean).is_err(), "mmap compact");
        assert!(PreadIndex::open(&path, Euclidean).is_err(), "pread");
        assert!(PreadIndex::open(&compact, Euclidean).is_err(), "pread compact");

        let mut labeled: LabeledIndex<Cosine, u32> =
            Builder::new().seed(801).build_labeled(Cosine).unwrap();
        labeled.insert(vec![1.0, 0.0], 7).unwrap();
        let labeled_path = dir.path().join("metric-labeled.hnsw");
        labeled.save(&labeled_path).expect("labeled save failed");
        assert!(LabeledIndex::<Euclidean, u32>::load(&labeled_path, Euclidean).is_err());
        assert!(
            LabeledIndex::<Euclidean, u32>::load_mmap(&labeled_path, Euclidean).is_err()
        );
        assert!(
            LabeledIndex::<Euclidean, u32>::load_mmap_fixed(&labeled_path, Euclidean)
                .is_err()
        );
        assert!(LabeledIndex::<Cosine, u32>::load(&labeled_path, Cosine).is_ok());
        assert!(PreadIndex::open(&compact, Cosine).is_ok());
    }

    /// Snapshots written before the header recorded a metric store zero there,
    /// and must keep loading with any metric.
    #[test]
    fn snapshots_without_a_recorded_metric_still_load() {
        use std::io::{Seek, SeekFrom, Write};

        let (index, _) = metric_sensitive_index();
        let dir = tempdir();
        let path = dir.path().join("metric-legacy.hnsw");
        persist::save(&index, &path).expect("save failed");

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        file.seek(SeekFrom::Start(72)).expect("seek failed");
        file.write_all(&0u32.to_le_bytes()).expect("write failed");
        drop(file);

        persist::load(&path, Euclidean).expect("legacy snapshot must still load");
        persist::load(&path, Cosine).expect("legacy snapshot must still load");
        persist::load_mmap(&path, Manhattan).expect("legacy snapshot must still load");
    }

    #[test]
    fn unidentified_custom_metrics_are_not_checked() {
        #[derive(Clone, Copy, Default)]
        struct Custom;
        impl Distance for Custom {
            fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
                Euclidean.distance(a, b)
            }
        }
        assert_eq!(Custom::metric_id(), 0, "custom metrics default to unspecified");

        let mut index: Hnsw<Custom> = Builder::new().seed(802).build(Custom).unwrap();
        index.insert(vec![1.0, 0.0]).unwrap();
        index.insert(vec![0.0, 1.0]).unwrap();
        let dir = tempdir();
        let path = dir.path().join("metric-custom.hnsw");
        persist::save(&index, &path).expect("save failed");

        // Wrote 0, so anything may open it.
        persist::load(&path, Custom).expect("custom load failed");
        persist::load(&path, Euclidean).expect("unidentified metric is not checked");

        // The reverse is refused: an unidentified metric cannot be shown to
        // match a file that did record one.
        let (cosine_index, _) = metric_sensitive_index();
        let identified = dir.path().join("metric-identified.hnsw");
        persist::save(&cosine_index, &identified).expect("save failed");
        let error = persist::load(&identified, Custom)
            .err()
            .expect("unidentified metric must not open an identified index");
        assert!(
            error.to_string().contains("Distance::metric_id"),
            "the error should say how to declare compatibility: {error}"
        );

        #[derive(Clone, Copy, Default)]
        struct CustomCosine;
        impl Distance for CustomCosine {
            fn metric_id() -> u32 { Cosine::metric_id() }
            fn distance(&self, a: &[f32], b: &[f32]) -> f32 { Cosine.distance(a, b) }
        }
        persist::load(&identified, CustomCosine)
            .expect("declaring the built-in's id should permit the load");
    }

    #[test]
    fn builtin_metric_ids_are_distinct_and_reserved() {
        use crate::distance::RESERVED_METRIC_IDS;
        let ids = [
            SquaredEuclidean::metric_id(),
            Euclidean::metric_id(),
            Cosine::metric_id(),
            distance::DotProduct::metric_id(),
            Manhattan::metric_id(),
        ];
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "built-in metric ids must be distinct");
        assert!(ids.iter().all(|id| *id != 0));
        assert!(ids.iter().all(|id| *id < RESERVED_METRIC_IDS));
    }

    // ── pread ─────────────────────────────────────────────────────────────

    #[test]
    fn pread_search_matches_the_resident_index() {
        use crate::pread::PreadIndex;

        let (index, _) = make_hnsw(500, 16, 700);
        let dir = tempdir();
        let path = dir.path().join("pread.hnsw");
        persist::save(&index, &path).expect("save failed");

        let on_disk = PreadIndex::open(&path, Euclidean).expect("pread open failed");
        assert_eq!(on_disk.len(), 500);
        assert_eq!(on_disk.dim(), Some(16));

        for seed in 0..8 {
            let query: Vec<f32> =
                (0..16).map(|d| ((seed * 16 + d) as f32) * 0.017).collect();
            let expected = index.search(&query, 10, 100).unwrap();
            let actual = on_disk.search(&query, 10, 100).expect("pread search failed");
            assert_eq!(actual, expected, "pread search diverged for seed {seed}");
        }
    }

    #[test]
    fn pread_reads_back_every_stored_vector() {
        use crate::pread::PreadIndex;

        let (index, _) = make_hnsw(120, 8, 701);
        let dir = tempdir();
        let path = dir.path().join("pread-vectors.hnsw");
        persist::save(&index, &path).expect("save failed");

        let on_disk = PreadIndex::open(&path, Euclidean).expect("pread open failed");
        for id in 0..index.len() {
            assert_eq!(
                on_disk.get_vector(id).expect("read failed"),
                index.get_vector(id).unwrap(),
                "vector {id} differs when read from disk"
            );
        }
        let mut buf = Vec::new();
        on_disk.vectors().read_into(7, &mut buf).expect("read_into failed");
        assert_eq!(buf, index.get_vector(7).unwrap());

        assert_eq!(
            on_disk.get_vector(120).expect_err("out-of-range id must fail").kind(),
            std::io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn pread_supports_filtered_search_and_compact_snapshots() {
        use crate::pread::PreadIndex;

        let (index, _) = make_hnsw(300, 8, 702);
        let dir = tempdir();
        let query = vec![0.4f32; 8];

        let path = dir.path().join("pread-filtered.hnsw");
        persist::save(&index, &path).expect("save failed");
        let on_disk = PreadIndex::open(&path, Euclidean).expect("open failed");
        assert_eq!(
            on_disk
                .search_filtered(&query, 5, 100, |id| id % 7 == 0)
                .expect("filtered failed"),
            index.search_filtered(&query, 5, 100, |id| id % 7 == 0).unwrap()
        );

        let compact_path = dir.path().join("pread-compact.hnsw");
        persist::save_compact(&index, &compact_path).expect("compact save failed");
        let compact =
            PreadIndex::open(&compact_path, Euclidean).expect("compact open failed");
        assert_eq!(compact.len(), index.len());
        assert_eq!(
            compact.search(&query, 10, 100).expect("compact search failed"),
            index.search(&query, 10, 100).unwrap()
        );
    }

    /// A truncated file must come back as an error from the search call.
    /// Through a mapping the same situation is a SIGBUS the process cannot
    /// catch — being able to handle it is the reason this path exists.
    #[test]
    fn pread_reports_truncation_as_an_error_not_a_fault() {
        use crate::pread::PreadIndex;

        let (index, _) = make_hnsw(300, 16, 703);
        let dir = tempdir();
        let path = dir.path().join("pread-truncated.hnsw");
        persist::save(&index, &path).expect("save failed");

        let on_disk = PreadIndex::open(&path, Euclidean).expect("open failed");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for truncate failed");
        file.set_len(256 + 16 * 4 * 2).expect("truncate failed");
        drop(file);

        let error = on_disk
            .search(&[0.5; 16], 10, 100)
            .expect_err("search over a truncated file must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn pread_rejects_a_mismatched_query_dimension() {
        use crate::pread::PreadIndex;

        let (index, _) = make_hnsw(50, 8, 704);
        let dir = tempdir();
        let path = dir.path().join("pread-dim.hnsw");
        persist::save(&index, &path).expect("save failed");
        let on_disk = PreadIndex::open(&path, Euclidean).expect("open failed");
        assert!(on_disk.search(&[0.5; 3], 1, 10).is_err());
    }

    /// One `File` shared by many threads is only safe because positional reads
    /// do not touch the file cursor.
    #[test]
    fn pread_index_supports_concurrent_queries() {
        use crate::pread::PreadIndex;
        use std::sync::Arc;

        let (index, _) = make_hnsw(400, 16, 705);
        let dir = tempdir();
        let path = dir.path().join("pread-threads.hnsw");
        persist::save(&index, &path).expect("save failed");

        let on_disk = Arc::new(PreadIndex::open(&path, Euclidean).expect("open failed"));
        let queries: Vec<Vec<f32>> = (0..16)
            .map(|q| (0..16).map(|d| ((q * 16 + d) as f32) * 0.013).collect())
            .collect();
        let expected: Vec<Vec<SearchResult>> = queries
            .iter()
            .map(|q| on_disk.search(q, 5, 50).expect("search failed"))
            .collect();

        let handles: Vec<_> = (0..6)
            .map(|_| {
                let on_disk = Arc::clone(&on_disk);
                let queries = queries.clone();
                std::thread::spawn(move || {
                    queries
                        .iter()
                        .map(|q| on_disk.search(q, 5, 50).expect("search failed"))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().expect("thread panicked"), expected);
        }
    }

    #[test]
    fn pread_graph_reports_non_resident_vectors() {
        use crate::pread::PreadIndex;

        let (index, _) = make_hnsw(30, 8, 706);
        let dir = tempdir();
        let path = dir.path().join("pread-noresident.hnsw");
        persist::save(&index, &path).expect("save failed");
        let on_disk = PreadIndex::open(&path, Euclidean).expect("open failed");

        assert_eq!(on_disk.graph().len(), 30);
        assert!(matches!(
            on_disk.graph().get_vector(0),
            Err(Error::VectorsNotResident)
        ));
    }

    // ── parallel build ────────────────────────────────────────────────────

    #[cfg(feature = "parallel")]
    mod parallel_build {
        use super::*;

        fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
            use rand::{RngExt, SeedableRng};
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            (0..n)
                .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
                .collect()
        }

        #[test]
        fn parallel_build_produces_a_usable_index() {
            let vectors = corpus(2_000, 16, 900);
            let index: Hnsw<Euclidean> = Builder::new()
                .m(16)
                .ef_construction(200)
                .seed(900)
                .build_parallel(Euclidean, vectors.clone())
                .unwrap();

            assert_eq!(index.len(), 2_000);
            assert_eq!(index.dim(), Some(16));
            for (id, vector) in vectors.iter().enumerate() {
                assert_eq!(index.get_vector(id).unwrap(), vector.as_slice());
            }
            for id in [0usize, 1, 999, 1_999] {
                assert_eq!(index.search(&vectors[id], 1, 100).unwrap()[0].id, id);
            }
        }

        /// The parallel path must produce a graph of comparable quality. The
        /// gap grows with dimension (see the `parallel` module docs), which is
        /// why this is a floor plus a tolerance, not strict parity.
        #[test]
        fn parallel_build_recall_matches_sequential() {
            let vectors = corpus(2_000, 32, 901);

            let mut sequential = Builder::new()
                .m(16)
                .ef_construction(200)
                .seed(901)
                .build(Euclidean)
                .unwrap();
            for vector in &vectors {
                sequential.insert(vector.clone()).unwrap();
            }
            let parallel: Hnsw<Euclidean> = Builder::new()
                .m(16)
                .ef_construction(200)
                .seed(901)
                .build_parallel(Euclidean, vectors.clone())
                .unwrap();

            let sequential_recall = recall(&sequential, &vectors, 10, 100, 50);
            let parallel_recall = recall(&parallel, &vectors, 10, 100, 50);
            assert!(parallel_recall >= 0.90, "parallel recall too low");
            assert!(parallel_recall + 0.05 >= sequential_recall);
        }

        #[test]
        fn parallel_build_graph_is_structurally_sound() {
            let vectors = corpus(1_000, 8, 902);
            let index: Hnsw<Euclidean> = Builder::new()
                .m(8)
                .ef_construction(100)
                .seed(902)
                .build_parallel(Euclidean, vectors)
                .unwrap();

            let m0 = index.config().m0();
            let m = index.config().m;
            for node in 0..index.graph.node_count() {
                for layer in 0..index.graph.level_count(node) {
                    let limit = if layer == 0 { m0 } else { m };
                    let count = index.graph.neighbour_count(node, layer);
                    assert!(count <= limit, "node {node} layer {layer} over limit");
                    for (neighbour, _) in
                        index.graph.neighbours(node, layer).into_iter().flatten()
                    {
                        assert_ne!(neighbour as usize, node, "self-loop at {node}");
                        assert!((neighbour as usize) < index.len());
                    }
                }
            }
            let (entry_id, entry_level) = index.entry_point.expect("entry point");
            assert_eq!(index.graph.level_count(entry_id) - 1, entry_level);
            for node in 0..index.graph.node_count() {
                assert!(index.graph.level_count(node) - 1 <= entry_level);
            }
        }

        #[test]
        fn parallel_build_round_trips_through_persistence() {
            let vectors = corpus(500, 8, 903);
            let index: Hnsw<Euclidean> = Builder::new()
                .seed(903)
                .build_parallel(Euclidean, vectors)
                .unwrap();
            let dir = tempdir();
            let path = dir.path().join("parallel.hnsw");
            persist::save(&index, &path).expect("save failed");
            let loaded = persist::load(&path, Euclidean).expect("load failed");
            assert_eq!(index.len(), loaded.len());
            let query = vec![0.5f32; 8];
            assert_eq!(
                index.search(&query, 10, 100).unwrap(),
                loaded.search(&query, 10, 100).unwrap()
            );
        }

        #[test]
        fn parallel_build_handles_degenerate_inputs() {
            let empty: Hnsw<Euclidean> = Builder::new()
                .seed(904)
                .build_parallel(Euclidean, Vec::new())
                .unwrap();
            assert!(empty.is_empty());
            assert!(empty.search(&[0.0], 1, 10).unwrap().is_empty());

            let single: Hnsw<Euclidean> = Builder::new()
                .seed(905)
                .build_parallel(Euclidean, vec![vec![1.0, 2.0]])
                .unwrap();
            assert_eq!(single.len(), 1);
            assert_eq!(single.search(&[1.0, 2.0], 1, 10).unwrap()[0].id, 0);

            let duplicates: Hnsw<Euclidean> = Builder::new()
                .seed(906)
                .build_parallel(Euclidean, vec![vec![3.0, 4.0]; 300])
                .unwrap();
            assert_eq!(duplicates.len(), 300);
            assert_eq!(duplicates.search(&[3.0, 4.0], 5, 50).unwrap().len(), 5);
        }

        #[test]
        fn parallel_build_rejects_ragged_input() {
            let result: crate::Result<Hnsw<Euclidean>> = Builder::new()
                .seed(907)
                .build_parallel(Euclidean, vec![vec![1.0, 2.0], vec![3.0]]);
            assert!(matches!(
                result,
                Err(Error::DimensionMismatch { expected: 2, actual: 1 })
            ));
        }
    }

    // ── concurrent search ─────────────────────────────────────────────────

    /// `Hnsw` is `Sync`, so many threads may query one index at once. Each
    /// needs its own workspace; the thread-local path must not share state.
    #[test]
    fn concurrent_searches_agree_with_sequential_results() {
        use std::sync::Arc;

        let index = Arc::new(build_index(1_000, 16, 910));
        let queries: Vec<Vec<f32>> = (0..32)
            .map(|q| (0..16).map(|d| ((q * 16 + d) as f32) * 0.01).collect())
            .collect();
        let expected: Vec<Vec<SearchResult>> = queries
            .iter()
            .map(|q| index.search(q, 10, 100).unwrap())
            .collect();

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let index = Arc::clone(&index);
                let queries = queries.clone();
                std::thread::spawn(move || {
                    queries
                        .iter()
                        .map(|q| index.search(q, 10, 100).unwrap())
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        for handle in handles {
            assert_eq!(handle.join().expect("search thread panicked"), expected);
        }
    }

    /// A filter predicate that itself searches re-enters the thread-local
    /// workspace; that must fall back to private storage, not deadlock.
    #[test]
    fn reentrant_search_from_a_filter_predicate_is_safe() {
        let outer = build_index(200, 8, 911);
        let inner = build_index(200, 8, 912);
        let query = [0.4f32; 8];
        let hits = outer
            .search_filtered(&query, 5, 50, |_id| {
                !inner.search(&query, 1, 10).unwrap().is_empty()
            })
            .unwrap();
        assert_eq!(hits.len(), 5);
    }

    // ── recall tests ──────────────────────────────────────────────────────

    fn recall(index: &Hnsw<Euclidean>, vectors: &[Vec<f32>], k: usize, ef: usize, n_queries: usize) -> f64 {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(99_999);
        let dim = vectors[0].len();

        let mut hits = 0usize;
        let mut total = 0usize;

        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let exact = exact_knn(vectors, &query, k);
            let approx: Vec<usize> = index.search(&query, k, ef).unwrap().iter().map(|r| r.id).collect();
            let exact_set: std::collections::HashSet<usize> = exact.into_iter().collect();
            for id in &approx {
                if exact_set.contains(id) {
                    hits += 1;
                }
            }
            total += k;
        }

        hits as f64 / total as f64
    }

    #[test]
    fn recall_128d_is_acceptable() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(77);
        let dim = 128;
        let n = 1_000;

        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut index = Builder::new()
            .m(16)
            .ef_construction(200)
            .seed(42)
            .build(Euclidean).unwrap();

        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v.clone()).unwrap();
            vectors.push(v);
        }

        let r = recall(&index, &vectors, 10, 100, 100);
        println!("Recall@10 (128d, 1k vectors, ef=100): {:.2}%", r * 100.0);
        // Expect ≥ 90 % recall with these parameters.
        assert!(r >= 0.90, "recall {:.2}% is too low", r * 100.0);
    }

    #[test]
    fn recall_32d_high_ef_is_near_perfect() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(55);
        let dim = 32;
        let n = 500;

        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut index = Builder::new()
            .m(32)
            .ef_construction(400)
            .seed(13)
            .build(Euclidean).unwrap();

        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v.clone()).unwrap();
            vectors.push(v);
        }

        let r = recall(&index, &vectors, 10, 500, 50);
        println!("Recall@10 (32d, 500 vectors, ef=500): {:.2}%", r * 100.0);
        assert!(r >= 0.98, "recall {:.2}% is too low", r * 100.0);
    }

    // ── distance metric tests ─────────────────────────────────────────────

    #[test]
    fn squared_euclidean_finds_correct_neighbour() {
        let mut index = Builder::new().seed(10).build(SquaredEuclidean).unwrap();
        index.insert(vec![0.0, 0.0]).unwrap(); // id=0
        index.insert(vec![1.0, 0.0]).unwrap(); // id=1
        index.insert(vec![5.0, 0.0]).unwrap(); // id=2
        let res = index.search(&[0.2, 0.0], 1, 10).unwrap();
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn cosine_distance_orthogonal_vectors() {
        let mut index = Builder::new().seed(11).build(Cosine).unwrap();
        index.insert(vec![1.0, 0.0]).unwrap(); // id=0
        index.insert(vec![0.0, 1.0]).unwrap(); // id=1  orthogonal
        index.insert(vec![0.9, 0.1]).unwrap(); // id=2  close to id=0
        let res = index.search(&[1.0, 0.0], 1, 10).unwrap();
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn manhattan_metric_correct_order() {
        let mut index = Builder::new().seed(12).build(Manhattan).unwrap();
        index.insert(vec![0.0]).unwrap();  // id=0, dist=1.0 from query 1.0
        index.insert(vec![10.0]).unwrap(); // id=1, dist=9.0 from query 1.0
        index.insert(vec![1.5]).unwrap();  // id=2, dist=0.5 from query 1.0
        let res = index.search(&[1.0], 1, 10).unwrap();
        assert_eq!(res[0].id, 2);
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn two_identical_vectors() {
        let mut index = Builder::new().seed(20).build(Euclidean).unwrap();
        index.insert(vec![1.0, 1.0]).unwrap(); // id=0
        index.insert(vec![1.0, 1.0]).unwrap(); // id=1  duplicate
        let res = index.search(&[1.0, 1.0], 2, 10).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].distance, 0.0);
        assert_eq!(res[1].distance, 0.0);
    }

    #[test]
    fn one_dimensional_vectors() {
        let mut index = Builder::new().seed(21).build(Euclidean).unwrap();
        for i in 0..50_u32 {
            index.insert(vec![i as f32]).unwrap();
        }
        let res = index.search(&[25.0], 3, 30).unwrap();
        let ids: Vec<usize> = res.iter().map(|r| r.id).collect();
        assert!(ids.contains(&25));
    }

    #[test]
    fn large_dimension_does_not_panic() {
        let mut index = Builder::new().m(8).ef_construction(50).seed(22).build(Euclidean).unwrap();
        let dim: usize = 1024;
        for i in 0..50_u32 {
            let v: Vec<f32> = (0..dim).map(|j| (i as usize + j) as f32).collect();
            index.insert(v).unwrap();
        }
        let query: Vec<f32> = vec![1.0; dim];
        let res = index.search(&query, 5, 20).unwrap();
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn simple_neighbour_selection_fallback() {
        let mut index = Builder::new()
            .m(16)
            .ef_construction(100)
            .heuristic(false) // use simple selection
            .seed(30)
            .build(Euclidean).unwrap();
        for i in 0..100_u32 {
            index.insert(vec![i as f32, 0.0]).unwrap();
        }
        let res = index.search(&[50.0, 0.0], 3, 30).unwrap();
        // Should include 50
        assert!(res.iter().any(|r| r.id == 50));
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn stats_are_consistent() {
        let index = build_index(500, 32, 50);
        let stats = index.stats();
        assert_eq!(stats.num_vectors, 500);
        // Layer 0 must contain all nodes.
        assert_eq!(stats.layer_counts[0], 500);
        // Edge count must be even (undirected).
        assert_eq!(stats.layer_edges[0] % 2, 0);
        println!("{}", stats);
    }

    // ── Persistence tests ─────────────────────────────────────────────────

    fn make_hnsw(n: usize, dim: usize, seed: u64) -> (Hnsw<Euclidean>, Vec<Vec<f32>>) {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed + 5_000);
        let mut index = Builder::new().m(16).ef_construction(200).seed(seed).build(Euclidean).unwrap();
        let mut corpus = Vec::with_capacity(n);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v.clone()).unwrap();
            corpus.push(v);
        }
        (index, corpus)
    }

    #[test]
    fn persist_save_load_round_trip() {
        let (orig, _) = make_hnsw(200, 16, 300);
        let dir = tempdir();
        let path = dir.path().join("test.hnsw");
        persist::save(&orig, &path).expect("save failed");

        let loaded = persist::load(&path, Euclidean).expect("load failed");
        assert_eq!(orig.len(), loaded.len());
        assert_eq!(orig.dim(), loaded.dim());
        // Vectors must be identical.
        for i in 0..orig.len() {
            assert_eq!(orig.get_vector(i).unwrap(), loaded.get_vector(i).unwrap(),
                       "vector {i} differs after load");
        }
        // Search results must be identical (same graph topology).
        let q = vec![0.5f32; 16];
        let r_orig   = orig.search(&q, 5, 50).unwrap();
        let r_loaded = loaded.search(&q, 5, 50).unwrap();
        assert_eq!(r_orig.len(), r_loaded.len());
        for (a, b) in r_orig.iter().zip(r_loaded.iter()) {
            assert_eq!(a.id, b.id, "search result id differs");
            assert!((a.distance - b.distance).abs() < 1e-6,
                    "distance differs: {} vs {}", a.distance, b.distance);
        }
    }

    #[test]
    fn persist_mmap_load_round_trip() {
        let (orig, _) = make_hnsw(200, 16, 301);
        let dir  = tempdir();
        let path = dir.path().join("mmap_test.hnsw");
        persist::save(&orig, &path).expect("save failed");

        let mmap = persist::load_mmap(&path, Euclidean).expect("mmap load failed");
        assert!(matches!(
            &mmap.graph,
            crate::hnsw::GraphStore::Mapped(_)
        ));
        assert_eq!(orig.len(), mmap.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get_vector(i).unwrap(), mmap.get_vector(i).unwrap(),
                       "mmap vector {i} differs");
        }
        let q = vec![0.3f32; 16];
        let r_orig = orig.search(&q, 5, 50).unwrap();
        let r_mmap = mmap.search(&q, 5, 50).unwrap();
        for (a, b) in r_orig.iter().zip(r_mmap.iter()) {
            assert_eq!(a.id, b.id);
        }
        let filtered_orig = orig.search_filtered(&q, 5, 200, |id| id % 7 == 0).unwrap();
        let filtered_mmap = mmap.search_filtered(&q, 5, 200, |id| id % 7 == 0).unwrap();
        assert_eq!(filtered_orig, filtered_mmap);
    }

    #[test]
    fn compact_snapshot_is_smaller_and_search_equivalent() {
        let (orig, _) = make_hnsw(200, 16, 303);
        let dir = tempdir();
        let v1_path = dir.path().join("full-v1.hnsw");
        let v2_path = dir.path().join("compact-v2.hnsw");
        persist::save(&orig, &v1_path).expect("v1 save failed");

        // Re-encoding an existing mmap snapshot is an important production
        // path: compacting does not require rebuilding the graph.
        let v1_mmap =
            persist::load_mmap(&v1_path, Euclidean).expect("v1 mmap load failed");
        persist::save_compact(&v1_mmap, &v2_path).expect("v2 save failed");

        let mut edge_count = 0;
        for node in 0..orig.graph.node_count() {
            for layer in 0..orig.graph.level_count(node) {
                edge_count += orig.graph.neighbour_count(node, layer);
            }
        }
        let v1_bytes = std::fs::metadata(&v1_path).expect("v1 metadata failed").len();
        let v2_bytes = std::fs::metadata(&v2_path).expect("v2 metadata failed").len();
        assert_eq!(v1_bytes - v2_bytes, edge_count as u64 * 4);

        let compact =
            persist::load_mmap(&v2_path, Euclidean).expect("v2 mmap load failed");
        assert_eq!(orig.len(), compact.len());
        for i in 0..orig.len() {
            assert_eq!(orig.get_vector(i).unwrap(), compact.get_vector(i).unwrap());
        }

        let query = vec![0.3f32; 16];
        assert_eq!(
            orig.search(&query, 10, 100).unwrap(),
            compact.search(&query, 10, 100).unwrap()
        );
        assert_eq!(
            orig.search_filtered(&query, 10, 200, |id| id % 7 == 0).unwrap(),
            compact.search_filtered(&query, 10, 200, |id| id % 7 == 0).unwrap()
        );

        let compact_labeled_path = dir.path().join("compact-labeled-v2.hnsw");
        let mut labeled = Builder::new().seed(303).build_labeled(Euclidean).unwrap();
        for id in 0..orig.len() {
            labeled.insert(orig.get_vector(id).unwrap().to_vec(), id as u32).unwrap();
        }
        labeled
            .save_compact(&compact_labeled_path)
            .expect("compact labeled save failed");
        let mapped = LabeledIndex::<Euclidean, u32>::load_mmap_fixed(
            &compact_labeled_path,
            Euclidean,
        )
        .expect("compact labeled mmap load failed");
        let expected = mapped.search(&query, 10, 100).expect("mapped search failed");
        let mut workspace = SearchWorkspace::default();
        let actual = mapped
            .search_with_workspace(&query, 10, 100, &mut workspace)
            .expect("workspace search failed");
        assert_eq!(
            actual.iter().map(|result| result.id).collect::<Vec<_>>(),
            expected.iter().map(|result| result.id).collect::<Vec<_>>()
        );
        let transformed = mapped
            .search_map_with_workspace(
                &query,
                10,
                100,
                &mut workspace,
                |result| (result.id, result.payload),
            )
            .expect("mapped result transform failed");
        assert_eq!(
            transformed,
            expected
                .iter()
                .map(|result| (result.id, result.payload))
                .collect::<Vec<_>>()
        );
        let expected = mapped
            .search_filtered(&query, 10, 200, |id, _| id % 7 == 0)
            .expect("mapped filtered search failed");
        let actual = mapped
            .search_filtered_with_workspace(
                &query,
                10,
                200,
                |id, _| id % 7 == 0,
                &mut workspace,
            )
            .expect("workspace filtered search failed");
        assert_eq!(
            actual.iter().map(|result| result.id).collect::<Vec<_>>(),
            expected.iter().map(|result| result.id).collect::<Vec<_>>()
        );
        let transformed = mapped
            .search_filtered_map_with_workspace(
                &query,
                10,
                200,
                |id, _| id % 7 == 0,
                &mut workspace,
                |result| (result.id, result.payload),
            )
            .expect("mapped filtered result transform failed");
        assert_eq!(
            transformed,
            expected
                .iter()
                .map(|result| (result.id, result.payload))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn compact_snapshot_rejects_owned_load() {
        let (index, _) = make_hnsw(20, 8, 304);
        let dir = tempdir();
        let path = dir.path().join("compact-owned.hnsw");
        persist::save_compact(&index, &path).expect("compact save failed");

        let error = persist::load(&path, Euclidean)
            .err()
            .expect("owned compact load should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("read-only"));
        assert!(error.to_string().contains("load_mmap"));
    }

    #[test]
    fn compact_snapshot_supports_variable_and_fixed_payloads() {
        let (index, _) = make_hnsw(30, 8, 305);
        let dir = tempdir();

        let string_path = dir.path().join("compact-strings.hnsw");
        let strings: Vec<String> = (0..index.len()).map(|id| format!("item-{id}")).collect();
        persist::save_compact_with_payload(&index, &strings, &string_path)
            .expect("compact variable payload save failed");
        let (string_index, loaded_strings) =
            persist::load_mmap_with_payload::<_, String>(&string_path, Euclidean)
                .expect("compact variable payload mmap load failed");
        assert_eq!(loaded_strings, strings);
        assert_eq!(string_index.len(), index.len());

        let fixed_path = dir.path().join("compact-u32.hnsw");
        let labels: Vec<u32> = (0..index.len() as u32).map(|id| id * 10).collect();
        persist::save_compact_with_payload(&index, &labels, &fixed_path)
            .expect("compact fixed payload save failed");
        let (fixed_index, mapped_labels) =
            persist::load_mmap_with_fixed_payload::<_, u32>(&fixed_path, Euclidean)
                .expect("compact fixed payload mmap load failed");
        assert_eq!(fixed_index.len(), index.len());
        for (id, expected) in labels.iter().copied().enumerate() {
            assert_eq!(mapped_labels.get(id).expect("payload decode failed"), expected);
        }
    }

    #[test]
    fn compact_snapshot_rejects_truncated_adjacency() {
        let (index, _) = make_hnsw(20, 8, 306);
        let dir = tempdir();
        let path = dir.path().join("compact-truncated.hnsw");
        persist::save_compact(&index, &path).expect("compact save failed");

        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        let len = file.metadata().expect("metadata failed").len();
        file.set_len(len - 17).expect("truncate failed");
        drop(file);

        let error = persist::load_mmap(&path, Euclidean)
            .err()
            .expect("truncated compact adjacency should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("connection list"));
    }

    #[test]
    fn compact_snapshot_rejects_out_of_range_neighbour() {
        use std::io::{Read, Seek, SeekFrom, Write};

        let n = 20;
        let dim = 8;
        let (index, _) = make_hnsw(n, dim, 307);
        let dir = tempdir();
        let path = dir.path().join("compact-invalid-neighbour.hnsw");
        persist::save_compact(&index, &path).expect("compact save failed");

        let offsets_start = 256 + n * dim * 4 + n * 4;
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open failed");
        file.seek(SeekFrom::Start(offsets_start as u64))
            .expect("seek failed");
        let mut offset = [0; 8];
        file.read_exact(&mut offset).expect("offset read failed");
        let first_record = u64::from_le_bytes(offset);
        file.seek(SeekFrom::Start(first_record))
            .expect("record seek failed");
        let mut count = [0; 4];
        file.read_exact(&mut count).expect("count read failed");
        assert!(u32::from_le_bytes(count) > 0);
        file.write_all(&(n as u32).to_le_bytes())
            .expect("neighbour write failed");
        drop(file);

        let error = persist::load_mmap(&path, Euclidean)
            .err()
            .expect("out-of-range neighbour should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("invalid neighbour id"));
    }

    #[test]
    fn compact_snapshot_rejects_out_of_range_entry_point() {
        use std::io::{Seek, SeekFrom, Write};

        let n = 20;
        let (index, _) = make_hnsw(n, 8, 308);
        let dir = tempdir();
        let path = dir.path().join("compact-invalid-entry-point.hnsw");
        persist::save_compact(&index, &path).expect("compact save failed");

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        file.seek(SeekFrom::Start(52)).expect("seek failed");
        file.write_all(&(n as u64).to_le_bytes())
            .expect("entry-point write failed");
        drop(file);

        let error = persist::load_mmap(&path, Euclidean)
            .err()
            .expect("out-of-range entry point should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("entry-point id"));
    }

    #[test]
    fn compact_snapshot_rejects_vector_size_overflow() {
        use std::io::{Seek, SeekFrom, Write};

        let (index, _) = make_hnsw(20, 8, 309);
        let dir = tempdir();
        let path = dir.path().join("compact-vector-overflow.hnsw");
        persist::save_compact(&index, &path).expect("compact save failed");

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        file.seek(SeekFrom::Start(20)).expect("seek failed");
        file.write_all(&u64::MAX.to_le_bytes())
            .expect("dimension write failed");
        drop(file);

        let error = persist::load_mmap(&path, Euclidean)
            .err()
            .expect("overflowing vector dimensions should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("vector"));
    }

    #[test]
    fn compact_snapshot_rejects_implausible_node_level() {
        use std::io::{Seek, SeekFrom, Write};

        let n = 20;
        let dim = 8;
        let (index, _) = make_hnsw(n, dim, 310);
        let dir = tempdir();
        let path = dir.path().join("compact-invalid-level.hnsw");
        persist::save_compact(&index, &path).expect("compact save failed");

        let levels_start = 256 + n * dim * 4;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        file.seek(SeekFrom::Start(levels_start as u64))
            .expect("seek failed");
        file.write_all(&u32::MAX.to_le_bytes())
            .expect("level write failed");
        drop(file);

        let error = persist::load_mmap(&path, Euclidean)
            .err()
            .expect("implausible node level should fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("supported maximum"));
    }

    #[test]
    fn persist_mmap_rejects_invalid_graph_offset() {
        use std::io::{Seek, SeekFrom, Write};

        let n = 20;
        let dim = 8;
        let (index, _) = make_hnsw(n, dim, 302);
        let dir = tempdir();
        let path = dir.path().join("bad_graph_offset.hnsw");
        persist::save(&index, &path).expect("save failed");

        let offsets_start = 256 + n * dim * 4 + n * 4;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        file.seek(SeekFrom::Start(offsets_start as u64))
            .expect("seek failed");
        file.write_all(&0u64.to_le_bytes()).expect("write failed");
        drop(file);

        let error = persist::load_mmap(&path, Euclidean)
            .err()
            .expect("invalid graph offset should be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("invalid connection offset"));
    }

    #[test]
    fn persist_empty_index() {
        let empty: Hnsw<Euclidean> = Builder::new().build(Euclidean).unwrap();
        let dir  = tempdir();
        let path = dir.path().join("empty.hnsw");
        persist::save(&empty, &path).expect("save empty failed");
        let loaded = persist::load(&path, Euclidean).expect("load empty failed");
        assert_eq!(loaded.len(), 0);
        assert!(loaded.search(&[0.0, 1.0], 5, 10).unwrap().is_empty());
    }

    // ── LabeledIndex tests ────────────────────────────────────────────────

    #[test]
    fn labeled_insert_and_search_u32() {
        let mut idx: LabeledIndex<Euclidean, u32> =
            Builder::new().seed(400).build_labeled(Euclidean).unwrap();
        idx.insert(vec![0.0, 0.0], 10_u32).unwrap();
        idx.insert(vec![1.0, 0.0], 20_u32).unwrap();
        idx.insert(vec![0.0, 1.0], 30_u32).unwrap();

        let hits = idx.search(&[0.1, 0.0], 1, 20).unwrap();
        assert_eq!(hits[0].payload, &10_u32);
        assert_eq!(hits[0].id, 0);

        let filtered = idx.search_filtered(&[0.1, 0.0], 2, 20, |_id, payload| *payload >= 20).unwrap();
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|result| *result.payload >= 20));
    }

    #[test]
    fn seeded_labeled_builder_is_byte_reproducible() {
        let build = || {
            let mut index = Builder::new()
                .m(8)
                .ef_construction(32)
                .seed(400)
                .build_labeled(Euclidean).unwrap();
            for id in 0..200_u32 {
                index.insert(vec![id as f32, (id % 17) as f32], id).unwrap();
            }
            index
        };
        let directory = tempdir();
        let left = directory.path().join("seeded-labeled-left.hnsw");
        let right = directory.path().join("seeded-labeled-right.hnsw");
        build().save_compact(&left).expect("left save failed");
        build().save_compact(&right).expect("right save failed");
        assert_eq!(
            std::fs::read(left).expect("left read failed"),
            std::fs::read(right).expect("right read failed")
        );
    }

    #[test]
    fn labeled_insert_and_search_string() {
        let mut idx: LabeledIndex<Euclidean, String> =
            Builder::new().seed(401).build_labeled(Euclidean).unwrap();
        idx.insert(vec![1.0, 0.0], "cat".to_string()).unwrap();
        idx.insert(vec![0.0, 1.0], "dog".to_string()).unwrap();
        idx.insert(vec![0.5, 0.5], "rabbit".to_string()).unwrap();

        let hits = idx.search(&[0.9, 0.1], 1, 20).unwrap();
        assert_eq!(hits[0].payload, "cat");
        assert_eq!(hits[0].embedding, &[1.0f32, 0.0]);
    }

    #[test]
    fn labeled_search_returns_embedding() {
        let mut idx: LabeledIndex<Euclidean, ()> =
            Builder::new().seed(402).build_labeled(Euclidean).unwrap();
        let v = vec![3.0f32, 4.0];
        idx.insert(v.clone(), ()).unwrap();
        let hits = idx.search(&[3.0, 4.0], 1, 10).unwrap();
        assert_eq!(hits[0].embedding, v.as_slice());
    }

    #[test]
    fn labeled_save_load_u32() {
        let mut idx: LabeledIndex<Euclidean, u32> =
            Builder::new().seed(410).build_labeled(Euclidean).unwrap();
        for i in 0..50_u32 {
            idx.insert(vec![i as f32, (i * 2) as f32], i * 10).unwrap();
        }
        let dir  = tempdir();
        let path = dir.path().join("labeled_u32.hnsw");
        idx.save(&path).expect("save failed");

        let loaded = LabeledIndex::<Euclidean, u32>::load(&path, Euclidean)
            .expect("load failed");
        assert_eq!(loaded.len(), 50);
        for i in 0..50_usize {
            assert_eq!(loaded.get_payload(i).unwrap(), &(i as u32 * 10));
            assert_eq!(loaded.get_embedding(i).unwrap(), &[i as f32, (i * 2) as f32]);
        }
        let hits = loaded.search(&[25.0, 50.0], 1, 30).unwrap();
        assert_eq!(hits[0].id, 25);
        assert_eq!(hits[0].payload, &250_u32);
    }

    #[test]
    fn labeled_save_load_string() {
        let labels = ["alpha", "beta", "gamma", "delta", "epsilon"];
        let mut idx: LabeledIndex<Euclidean, String> =
            Builder::new().seed(411).build_labeled(Euclidean).unwrap();
        for (i, &s) in labels.iter().enumerate() {
            idx.insert(vec![i as f32], s.to_string()).unwrap();
        }
        let dir  = tempdir();
        let path = dir.path().join("labeled_str.hnsw");
        idx.save(&path).expect("save failed");

        let loaded = LabeledIndex::<Euclidean, String>::load(&path, Euclidean)
            .expect("load failed");
        for (i, &s) in labels.iter().enumerate() {
            assert_eq!(loaded.get_payload(i).unwrap(), s);
        }
    }

    #[test]
    fn labeled_save_load_vec_f32_payload() {
        // Payload is a secondary embedding (variable-width)
        let mut idx: LabeledIndex<Euclidean, Vec<f32>> =
            Builder::new().seed(412).build_labeled(Euclidean).unwrap();
        let primary = vec![1.0f32, 0.0];
        let secondary = vec![0.0f32, 0.0, 1.0]; // different dim
        idx.insert(primary.clone(), secondary.clone()).unwrap();
        let dir  = tempdir();
        let path = dir.path().join("labeled_vecf32.hnsw");
        idx.save(&path).expect("save failed");

        let loaded = LabeledIndex::<Euclidean, Vec<f32>>::load(&path, Euclidean)
            .expect("load failed");
        assert_eq!(loaded.get_payload(0).unwrap(), &secondary);
    }

    #[test]
    fn labeled_mmap_load() {
        let mut idx: LabeledIndex<Euclidean, u32> =
            Builder::new().seed(420).build_labeled(Euclidean).unwrap();
        for i in 0..30_u32 {
            idx.insert(vec![i as f32], i).unwrap();
        }
        let dir  = tempdir();
        let path = dir.path().join("labeled_mmap.hnsw");
        idx.save(&path).expect("save failed");

        let mmap = LabeledIndex::<Euclidean, u32>::load_mmap(&path, Euclidean)
            .expect("mmap load failed");
        assert_eq!(mmap.len(), 30);
        for i in 0..30_usize {
            assert_eq!(mmap.get_payload(i).unwrap(), &(i as u32));
        }
    }

    #[test]
    fn labeled_fixed_payloads_stay_mapped() {
        let mut idx: LabeledIndex<Euclidean, u32> =
            Builder::new().seed(421).build_labeled(Euclidean).unwrap();
        for i in 0..30_u32 {
            idx.insert(vec![i as f32], i * 10).unwrap();
        }
        let dir = tempdir();
        let path = dir.path().join("labeled_fixed_mmap.hnsw");
        idx.save(&path).expect("save failed");

        let mmap = LabeledIndex::<Euclidean, u32>::load_mmap_fixed(&path, Euclidean)
            .expect("fixed mmap load failed");
        assert!(matches!(
            &mmap.inner.graph,
            crate::hnsw::GraphStore::Mapped(_)
        ));
        assert_eq!(mmap.len(), 30);
        for i in 0..30_usize {
            assert_eq!(mmap.get_payload(i).expect("payload decode failed"), i as u32 * 10);
        }
        assert_eq!(
            mmap.get_payload(30).expect_err("out-of-range id should fail").kind(),
            std::io::ErrorKind::InvalidInput
        );

        let hits = mmap.search(&[12.0], 1, 20).expect("search failed");
        assert_eq!(hits[0].id, 12);
        assert_eq!(hits[0].payload, 120);
        assert_eq!(hits[0].embedding, &[12.0]);

        let filtered = mmap
            .search_filtered(&[12.0], 3, 30, |_id, payload| *payload % 40 == 0)
            .expect("filtered search failed");
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0].id, 12);
        assert!(filtered.iter().all(|result| result.payload % 40 == 0));
    }

    #[test]
    fn labeled_mapped_payloads_reject_variable_width_types() {
        let mut idx: LabeledIndex<Euclidean, String> =
            Builder::new().seed(422).build_labeled(Euclidean).unwrap();
        idx.insert(vec![1.0], "one".to_string()).unwrap();
        let dir = tempdir();
        let path = dir.path().join("labeled_variable_mmap.hnsw");
        idx.save(&path).expect("save failed");

        let error = LabeledIndex::<Euclidean, String>::load_mmap_fixed(&path, Euclidean)
            .err()
            .expect("variable-width mapped payload should be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn labeled_mapped_payloads_reject_truncated_columns() {
        let mut idx: LabeledIndex<Euclidean, u64> =
            Builder::new().seed(423).build_labeled(Euclidean).unwrap();
        idx.insert(vec![1.0], 10).unwrap();
        idx.insert(vec![2.0], 20).unwrap();
        let dir = tempdir();
        let path = dir.path().join("labeled_truncated_mmap.hnsw");
        idx.save(&path).expect("save failed");

        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open failed");
        let truncated_len = file.metadata().expect("metadata failed").len() - 1;
        file.set_len(truncated_len).expect("truncate failed");
        drop(file);

        let error = LabeledIndex::<Euclidean, u64>::load_mmap_fixed(&path, Euclidean)
            .err()
            .expect("truncated mapped payload should be rejected");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("payload column"));
    }

    // ── PairedIndex tests ─────────────────────────────────────────────────

    #[test]
    fn paired_insert_and_search_both_sides() {
        let mut idx: PairedIndex<Euclidean, Euclidean> = Builder::new()
            .m(16).ef_construction(50).seed(500)
            .build_paired(Euclidean, Euclidean).unwrap();

        // Three items: each has a 2-D A-embedding and 3-D B-embedding.
        idx.insert(vec![1.0, 0.0],       vec![0.9, 0.1, 0.0]).unwrap();   // id=0
        idx.insert(vec![0.0, 1.0],       vec![0.1, 0.8, 0.1]).unwrap();   // id=1
        idx.insert(vec![0.5, 0.5],       vec![0.3, 0.3, 0.4]).unwrap();   // id=2

        // Search A-space: query near item 0
        let hits_a = idx.search_by_a(&[0.9, 0.1], 1, 20).unwrap();
        assert_eq!(hits_a[0].id, 0);
        assert_eq!(hits_a[0].emb_b, &[0.9f32, 0.1, 0.0]);

        // Search B-space: query near item 1
        let hits_b = idx.search_by_b(&[0.1, 0.9, 0.0], 1, 20).unwrap();
        assert_eq!(hits_b[0].id, 1);
        assert_eq!(hits_b[0].emb_a, &[0.0f32, 1.0]);
    }

    #[test]
    fn seeded_paired_builder_is_byte_reproducible() {
        let build = || {
            let mut index = Builder::new()
                .m(8)
                .ef_construction(32)
                .seed(500)
                .build_paired(Euclidean, Euclidean).unwrap();
            for id in 0..200_u32 {
                index.insert(
                    vec![id as f32, (id % 11) as f32],
                    vec![(id % 13) as f32, id as f32],
                ).unwrap();
            }
            index
        };
        let directory = tempdir();
        let left = directory.path().join("seeded-paired-left");
        let right = directory.path().join("seeded-paired-right");
        build().save(&left).expect("left save failed");
        build().save(&right).expect("right save failed");
        for side in ["_a.hnsw", "_b.hnsw"] {
            let left = std::path::PathBuf::from(format!("{}{side}", left.display()));
            let right = std::path::PathBuf::from(format!("{}{side}", right.display()));
            assert_eq!(
                std::fs::read(left).expect("left read failed"),
                std::fs::read(right).expect("right read failed")
            );
        }
    }

    #[test]
    fn paired_len_consistent() {
        let mut idx: PairedIndex<Euclidean, Euclidean> =
            PairedIndex::new(Default::default(), Euclidean, Default::default(), Euclidean)
                .unwrap();
        assert_eq!(idx.len(), 0);
        for i in 0..10_u32 {
            idx.insert(vec![i as f32], vec![i as f32, i as f32]).unwrap();
            assert_eq!(idx.len(), i as usize + 1);
        }
    }

    #[test]
    fn paired_cross_side_retrieval() {
        let mut idx: PairedIndex<Euclidean, Euclidean> = Builder::new()
            .m(16).ef_construction(100).seed(501)
            .build_paired(Euclidean, Euclidean).unwrap();
        // 20 items
        for i in 0..20_u32 {
            idx.insert(vec![i as f32, 0.0], vec![0.0, i as f32]).unwrap();
        }
        // Search by A near item 10 → get B embedding of item 10
        let hits = idx.search_by_a(&[10.0, 0.0], 1, 30).unwrap();
        assert_eq!(hits[0].id, 10);
        assert_eq!(hits[0].emb_b, &[0.0f32, 10.0]);
        // Confirm: searching by B near item 10 → get A embedding of item 10
        let hits2 = idx.search_by_b(&[0.0, 10.0], 1, 30).unwrap();
        assert_eq!(hits2[0].id, 10);
        assert_eq!(hits2[0].emb_a, &[10.0f32, 0.0]);
    }

    #[test]
    fn paired_save_load() {
        let mut idx: PairedIndex<Euclidean, Euclidean> = Builder::new()
            .m(16).ef_construction(100).seed(510)
            .build_paired(Euclidean, Euclidean).unwrap();
        for i in 0..50_u32 {
            idx.insert(vec![i as f32], vec![i as f32, i as f32]).unwrap();
        }
        let dir = tempdir();
        let base = dir.path().join("paired");
        idx.save(&base).expect("save failed");

        let loaded = PairedIndex::<Euclidean, Euclidean>::load(&base, Euclidean, Euclidean)
            .expect("load failed");
        assert_eq!(loaded.len(), 50);
        for i in 0..50_usize {
            assert_eq!(loaded.get_emb_a(i).unwrap(), &[i as f32][..]);
            assert_eq!(loaded.get_emb_b(i).unwrap(), &[i as f32, i as f32][..]);
        }
        let hits = loaded.search_by_a(&[25.0], 1, 30).unwrap();
        assert_eq!(hits[0].id, 25);
    }

    #[test]
    fn paired_mmap_load() {
        let mut idx: PairedIndex<Euclidean, Euclidean> = Builder::new()
            .seed(520).build_paired(Euclidean, Euclidean).unwrap();
        for i in 0..30_u32 {
            idx.insert(vec![i as f32, 0.0], vec![0.0, i as f32, 1.0]).unwrap();
        }
        let dir  = tempdir();
        let base = dir.path().join("paired_mmap");
        idx.save(&base).expect("save failed");

        let m = PairedIndex::<Euclidean, Euclidean>::load_mmap(&base, Euclidean, Euclidean)
            .expect("mmap load failed");
        assert_eq!(m.len(), 30);
        // Spot-check a few vectors
        for i in [0, 15, 29] {
            assert_eq!(m.get_emb_a(i).unwrap(), &[i as f32, 0.0f32][..]);
            assert_eq!(m.get_emb_b(i).unwrap(), &[0.0f32, i as f32, 1.0][..]);
        }
    }

    /// A temp directory that is removed when the returned handle drops at the
    /// end of the test.
    ///
    /// The previous helper derived its name from `subsec_nanos()` alone, so two
    /// tests scheduled in the same nanosecond bucket shared a directory, and
    /// nothing ever deleted them.
    fn tempdir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("hnsw_test_")
            .tempdir()
            .expect("failed to create temp dir")
    }

    // ── PruneStrategy tests ───────────────────────────────────────────────

    /// Helper: build an index with a given prune strategy and return
    /// (index, corpus) so callers can run recall checks.
    fn build_with_prune(n: usize, dim: usize, seed: u64, ps: PruneStrategy)
        -> (Hnsw<Euclidean>, Vec<Vec<f32>>)
    {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed + 2_000);
        let mut index = Builder::new()
            .m(16)
            .ef_construction(200)
            .prune_strategy(ps)
            .seed(seed)
            .build(Euclidean).unwrap();
        let mut corpus = Vec::with_capacity(n);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            index.insert(v.clone()).unwrap();
            corpus.push(v);
        }
        (index, corpus)
    }

    #[test]
    fn prune_strategy_default_is_simple() {
        // Ensure Config::default() picks Simple so users get the fastest
        // behaviour out of the box without any builder call.
        assert_eq!(Config::default().prune_strategy, PruneStrategy::Simple);
        // Building via Builder without calling .prune_strategy() must also
        // default to Simple.
        let mut index = Builder::new().seed(0).build(Euclidean).unwrap();
        index.insert(vec![1.0, 2.0]).unwrap();
        // The index built successfully — no panics, correct result.
        assert_eq!(index.search(&[1.0, 2.0], 1, 10).unwrap()[0].id, 0);
    }

    #[test]
    fn prune_strategy_simple_gives_acceptable_recall() {
        let (index, corpus) = build_with_prune(500, 32, 101, PruneStrategy::Simple);
        let r = recall(&index, &corpus, 10, 200, 50);
        println!("Simple recall@10 (32d 500v ef=200): {:.2}%", r * 100.0);
        assert!(r >= 0.95, "Simple recall {:.2}% too low", r * 100.0);
    }

    #[test]
    fn prune_strategy_heuristic_gives_acceptable_recall() {
        let (index, corpus) = build_with_prune(500, 32, 101, PruneStrategy::Heuristic);
        let r = recall(&index, &corpus, 10, 200, 50);
        println!("Heuristic recall@10 (32d 500v ef=200): {:.2}%", r * 100.0);
        assert!(r >= 0.95, "Heuristic recall {:.2}% too low", r * 100.0);
    }

    #[test]
    fn prune_strategy_heuristic_recall_ge_simple() {
        // Heuristic must not be worse than Simple (it does strictly more work
        // to preserve diversity).  Run both on the same data and seed.
        let (idx_s, corpus) = build_with_prune(500, 128, 202, PruneStrategy::Simple);
        let (idx_h, _)      = build_with_prune(500, 128, 202, PruneStrategy::Heuristic);
        let r_s = recall(&idx_s, &corpus, 10, 100, 50);
        let r_h = recall(&idx_h, &corpus, 10, 100, 50);
        println!("Simple {:.2}%  Heuristic {:.2}%", r_s * 100.0, r_h * 100.0);
        // Allow up to 1 pp slack for statistical noise in the random queries.
        assert!(r_h + 0.01 >= r_s,
            "Heuristic recall ({:.2}%) should be ≥ Simple ({:.2}%)",
            r_h * 100.0, r_s * 100.0);
    }

    #[test]
    fn max_level_grows_with_more_inserts() {
        let index_small = build_index(10, 4, 60);
        let index_large = build_index(10_000, 4, 60);
        // With many more nodes the entry-point level is likely higher.
        // This is probabilistic but almost certain with 10 000 vs 10 nodes.
        let l_small = index_small.max_level().unwrap_or(0);
        let l_large = index_large.max_level().unwrap_or(0);
        println!("small max_level={l_small}, large max_level={l_large}");
        assert!(l_large >= l_small);
    }
}
