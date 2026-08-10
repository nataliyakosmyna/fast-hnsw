//! Vectorised distance kernels.
//!
//! Every kernel here computes exactly what the portable fold computes, only
//! faster; none of them trade accuracy for speed.  Each has a scalar
//! `*_portable` reference, and the test suite asserts the vector path agrees
//! with it across every length class. AArch64 additionally preserves the
//! portable fold bit-for-bit so enabling SIMD cannot silently steer HNSW
//! construction toward a different graph.
//!
//! # Why hand-written intrinsics at all
//!
//! The portable folds already auto-vectorise: LLVM turns the eight-accumulator
//! L2 fold into `ldp q…` + `fsub.4s` + `fmul.4s` on AArch64, processing eight
//! floats per iteration. Writing the intrinsics by hand still helps because it
//! guarantees vector loads and arithmetic without relying on loop recognition,
//! and — for L1 — uses AArch64's single-instruction absolute difference `fabd`.
//! The NEON kernels deliberately mirror the portable fold's eight logical
//! lanes, separate multiply/add, and final reduction order. That costs some
//! peak micro-kernel throughput but prevents last-bit distance changes from
//! producing a different, slower HNSW graph.
//!
//! Measured against the auto-vectorised fold, NEON on Apple silicon:
//!
//! | dim | L2 auto | L2 NEON |       | L1 auto | L1 NEON |       |
//! |-----|---------|---------|-------|---------|---------|-------|
//! | 128 | 13.8 ns | 8.8 ns  | 1.57x | 11.6 ns | 8.6 ns  | 1.35x |
//! | 768 | 80.4 ns | 59.6 ns | 1.35x | 68.8 ns | 55.4 ns | 1.24x |
//!
//! AVX2 gains more, because the portable fold has further to fall on x86 --
//! measured on an i9-12900H and a Ryzen 9 8945HS (L2, portable -> AVX2):
//!
//! | dim  | Intel portable | Intel AVX2 |       | AMD portable | AMD AVX2 |       |
//! |------|----------------|-----------|-------|--------------|----------|-------|
//! | 128  | 21.2 ns        | 6.9 ns    | 3.07x | 18.9 ns      | 6.0 ns   | 3.15x |
//! | 768  | 117.3 ns       | 35.2 ns   | 3.33x | 97.9 ns      | 31.5 ns  | 3.11x |
//! | 1536 | 233.8 ns       | 75.2 ns   | 3.11x | 193.4 ns     | 65.2 ns  | 2.97x |
//!
//! This matters most for L2 and L1 specifically, because those are the metrics
//! the `blas` feature deliberately cannot accelerate (see the `distance` module).
//!
//! # AVX-512
//!
//! The optional `avx512` feature adds 512-bit kernels, preferred at runtime
//! when the CPU reports `avx512f`. The return is modest — measured on a Zen 4
//! Ryzen 9 8945HS, L2 distance, median of three runs:
//!
//! | dim | AVX2    | AVX-512 |        |
//! |-----|---------|---------|--------|
//! | 64  | 4.4 ns  | 4.5 ns  | parity |
//! | 128 | 6.0 ns  | 5.8 ns  | parity |
//! | 256 | 10.6 ns | 9.9 ns  | ~7%    |
//! | 512 | 22.1 ns | 19.0 ns | ~14%   |
//! | 768 | 32.5 ns | 28.3 ns | ~13%   |
//!
//! So it is worth enabling for wide embeddings and not much else — and it
//! raises the effective MSRV to 1.89, since that is where the intrinsics
//! stabilised. Off by default for both reasons.
//!
//! (Individual runs on that machine produced occasional 3x outliers in either
//! direction; the table is the stable median. Treat single-shot kernel
//! measurements there with suspicion.)
//!
//! # Dispatch
//!
//! Gated on the `simd` feature, which is **on by default**; turning it off
//! selects the portable folds everywhere, which LLVM still auto-vectorises
//! (measurably slower, but with no `unsafe` and no architecture-specific code).
//!
//! * **AArch64** — NEON is architecturally guaranteed, so the kernels are
//!   selected at compile time with no runtime check.
//! * **x86-64** — AVX-512F (with the `avx512` feature), else AVX2 + FMA,
//!   detected at runtime via
//!   [`std::arch::is_x86_feature_detected`], which caches its answer in a
//!   static after the first call.  Machines without them fall back to the
//!   portable fold, which LLVM still vectorises to SSE2.
//! * **Everything else** — the portable fold.

// ─── Portable reference implementations ──────────────────────────────────────

/// Number of independent accumulators used by the portable folds.
///
/// Float addition is not associative, so the compiler may not re-order a plain
/// `zip().map().sum()` fold — it becomes one serial dependency chain whose
/// length is the vector dimension.  Splitting the fold across `LANES`
/// independent accumulators breaks that chain and lets the vectorizer work.
///
/// Unused in non-test builds on AArch64, where the NEON path always wins the
/// dispatch; it is still the fallback everywhere else.
#[allow(dead_code)]
const LANES: usize = 8;

macro_rules! portable_fold {
    ($name:ident, $term:expr) => {
        /// Portable reference kernel: folds the common prefix of `a` and `b`
        /// into [`LANES`] independent accumulators.
        ///
        /// Deliberately uses separate multiply and add rather than
        /// [`f32::mul_add`]: baseline `x86-64` has no FMA instruction, so
        /// `mul_add` there lowers to a libm call and is far slower.
        ///
        /// Serves two roles: the fallback on architectures with no vector
        /// path, and the reference the tests check the vector kernels against.
        /// On AArch64 the NEON path always wins the dispatch, so outside tests
        /// this is deliberately unused there.
        #[allow(dead_code)]
        #[inline(always)]
        pub(crate) fn $name(a: &[f32], b: &[f32]) -> f32 {
            // Trim to the common prefix so both remainders cover the same
            // element range, preserving `zip` semantics for unequal lengths.
            let common = a.len().min(b.len());
            let (a, b) = (&a[..common], &b[..common]);

            let mut sums = [0.0f32; LANES];
            let mut a_chunks = a.chunks_exact(LANES);
            let mut b_chunks = b.chunks_exact(LANES);
            for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
                for lane in 0..LANES {
                    let term: fn(f32, f32) -> f32 = $term;
                    sums[lane] += term(x[lane], y[lane]);
                }
            }
            let mut sum = ((sums[0] + sums[1]) + (sums[2] + sums[3]))
                + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
            for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
                let term: fn(f32, f32) -> f32 = $term;
                sum += term(*x, *y);
            }
            sum
        }
    };
}

portable_fold!(l2_squared_portable, |x, y| (x - y) * (x - y));
portable_fold!(l1_portable, |x, y| (x - y).abs());
portable_fold!(dot_portable, |x, y| x * y);

// ─── AArch64 / NEON ──────────────────────────────────────────────────────────

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod neon {
    use std::arch::aarch64::*;

    /// Reduce the portable fold's eight logical lanes in the same order.
    ///
    /// # Safety
    /// Both arguments are initialized NEON values and `lanes` has room for
    /// both four-float stores.
    #[inline(always)]
    unsafe fn reduce_eight_lanes(acc0: float32x4_t, acc1: float32x4_t) -> f32 {
        let mut lanes = [0.0_f32; 8];
        vst1q_f32(lanes.as_mut_ptr(), acc0);
        vst1q_f32(lanes.as_mut_ptr().add(4), acc1);
        ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
    }

    /// Sum of squared differences over the common prefix.
    ///
    /// # Safety
    /// NEON is architecturally mandatory on AArch64, so no feature detection is
    /// required. Every load reads 4 floats from an index the loop bound has
    /// already proven is at least 4 short of `n`, and `n` is the *shorter* of
    /// the two slice lengths, so all reads are in bounds for both.
    #[inline]
    pub(crate) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); 2];
            let mut i = 0;
            while i + 8 <= n {
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let offset = i + lane * 4;
                    let d = vsubq_f32(vld1q_f32(pa.add(offset)), vld1q_f32(pb.add(offset)));
                    *acc = vaddq_f32(*acc, vmulq_f32(d, d));
                }
                i += 8;
            }
            let mut sum = reduce_eight_lanes(acc[0], acc[1]);
            while i < n {
                let d = *pa.add(i) - *pb.add(i);
                sum += d * d;
                i += 1;
            }
            sum
        }
    }

    /// Sum of absolute differences, using NEON's single-instruction `fabd`.
    ///
    /// # Safety
    /// Same bounds argument as [`l2_squared`].
    #[inline]
    pub(crate) fn l1(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        unsafe {
            let (mut acc0, mut acc1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            let mut i = 0;
            while i + 8 <= n {
                acc0 = vaddq_f32(
                    acc0,
                    vabdq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i))),
                );
                acc1 = vaddq_f32(
                    acc1,
                    vabdq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4))),
                );
                i += 8;
            }
            let mut sum = reduce_eight_lanes(acc0, acc1);
            while i < n {
                sum += (*pa.add(i) - *pb.add(i)).abs();
                i += 1;
            }
            sum
        }
    }

    /// Dot product over the common prefix.
    ///
    /// # Safety
    /// Same bounds argument as [`l2_squared`].
    #[inline]
    pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        unsafe {
            // Match the portable fold's eight logical lanes and reduction
            // order. HNSW construction is comparison-driven, so changing the
            // last bit of a distance can select a different graph even when
            // the kernel is numerically equivalent within a tolerance.
            let mut acc = [vdupq_n_f32(0.0); 2];
            let mut i = 0;
            while i + 8 <= n {
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let offset = i + lane * 4;
                    *acc = vaddq_f32(
                        *acc,
                        vmulq_f32(vld1q_f32(pa.add(offset)), vld1q_f32(pb.add(offset))),
                    );
                }
                i += 8;
            }
            let mut sum = reduce_eight_lanes(acc[0], acc[1]);
            while i < n {
                sum += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            sum
        }
    }
}

// ─── x86-64 / AVX2 + FMA ─────────────────────────────────────────────────────

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontal sum of a 256-bit lane.
    ///
    /// # Safety
    /// Caller must have verified AVX2 support.
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum(v: __m256) -> f32 {
        let low = _mm256_castps256_ps128(v);
        let high = _mm256_extractf128_ps(v, 1);
        let sum = _mm_add_ps(low, high);
        let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        let sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
        _mm_cvtss_f32(sum)
    }

    /// Sum of squared differences over the common prefix.
    ///
    /// # Safety
    /// Caller must have verified AVX2 and FMA support. Every load reads 8
    /// floats from an index the loop bound has proven is at least 8 short of
    /// `n`, and `n` is the shorter of the two lengths, so all reads are in
    /// bounds for both slices. `_mm256_loadu_ps` has no alignment requirement.
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 2];
        let mut i = 0;
        while i + 16 <= n {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let offset = i + lane * 8;
                let d = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(offset)),
                    _mm256_loadu_ps(pb.add(offset)),
                );
                *acc = _mm256_fmadd_ps(d, d, *acc);
            }
            i += 16;
        }
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
            i += 8;
        }
        let mut sum = horizontal_sum(_mm256_add_ps(acc[0], acc[1]));
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            sum += d * d;
            i += 1;
        }
        sum
    }

    /// Sum of absolute differences.  AVX2 has no float `fabs`, so the sign bit
    /// is masked off directly.
    ///
    /// # Safety
    /// Same requirements and bounds argument as [`l2_squared`].
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn l1(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let mut acc = [_mm256_setzero_ps(); 2];
        let mut i = 0;
        while i + 16 <= n {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let offset = i + lane * 8;
                let d = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(offset)),
                    _mm256_loadu_ps(pb.add(offset)),
                );
                *acc = _mm256_add_ps(*acc, _mm256_and_ps(d, sign_mask));
            }
            i += 16;
        }
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc[0] = _mm256_add_ps(acc[0], _mm256_and_ps(d, sign_mask));
            i += 8;
        }
        let mut sum = horizontal_sum(_mm256_add_ps(acc[0], acc[1]));
        while i < n {
            sum += (*pa.add(i) - *pb.add(i)).abs();
            i += 1;
        }
        sum
    }

    /// Dot product over the common prefix.
    ///
    /// # Safety
    /// Same requirements and bounds argument as [`l2_squared`].
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 2];
        let mut i = 0;
        while i + 16 <= n {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let offset = i + lane * 8;
                *acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(offset)),
                    _mm256_loadu_ps(pb.add(offset)),
                    *acc,
                );
            }
            i += 16;
        }
        while i + 8 <= n {
            acc[0] = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i)),
                _mm256_loadu_ps(pb.add(i)),
                acc[0],
            );
            i += 8;
        }
        let mut sum = horizontal_sum(_mm256_add_ps(acc[0], acc[1]));
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }

    /// Whether this CPU supports the instructions the kernels above need.
    ///
    /// `is_x86_feature_detected!` caches its result in a static, so repeated
    /// calls cost a relaxed atomic load rather than a `cpuid`.
    #[inline(always)]
    pub(crate) fn available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
}


// ─── x86-64 / AVX-512 ────────────────────────────────────────────────────────

#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
// The AVX-512 intrinsics stabilised in Rust 1.89, above this crate's declared
// 1.85 MSRV. That is why the kernels are behind an opt-in feature: enabling it
// raises your effective minimum, and the crate's baseline stays where it is.
#[allow(clippy::incompatible_msrv)]
mod avx512 {
    use std::arch::x86_64::*;

    /// Sum of squared differences over the common prefix.
    ///
    /// # Safety
    /// Caller must have verified AVX-512F support. Every load reads 16 floats
    /// from an index the loop bound has proven is at least 16 short of `n`, and
    /// `n` is the shorter of the two lengths, so all reads are in bounds for
    /// both slices. `_mm512_loadu_ps` has no alignment requirement.
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm512_setzero_ps(); 2];
        let mut i = 0;
        while i + 32 <= n {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let offset = i + lane * 16;
                let d = _mm512_sub_ps(
                    _mm512_loadu_ps(pa.add(offset)),
                    _mm512_loadu_ps(pb.add(offset)),
                );
                *acc = _mm512_fmadd_ps(d, d, *acc);
            }
            i += 32;
        }
        while i + 16 <= n {
            let d = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
            acc[0] = _mm512_fmadd_ps(d, d, acc[0]);
            i += 16;
        }
        let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(acc[0], acc[1]));
        while i < n {
            let d = *pa.add(i) - *pb.add(i);
            sum += d * d;
            i += 1;
        }
        sum
    }

    /// Sum of absolute differences.
    ///
    /// # Safety
    /// Same requirements and bounds argument as [`l2_squared`].
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn l1(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm512_setzero_ps(); 2];
        let mut i = 0;
        while i + 32 <= n {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let offset = i + lane * 16;
                let d = _mm512_sub_ps(
                    _mm512_loadu_ps(pa.add(offset)),
                    _mm512_loadu_ps(pb.add(offset)),
                );
                *acc = _mm512_add_ps(*acc, _mm512_abs_ps(d));
            }
            i += 32;
        }
        while i + 16 <= n {
            let d = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
            acc[0] = _mm512_add_ps(acc[0], _mm512_abs_ps(d));
            i += 16;
        }
        let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(acc[0], acc[1]));
        while i < n {
            sum += (*pa.add(i) - *pb.add(i)).abs();
            i += 1;
        }
        sum
    }

    /// Dot product over the common prefix.
    ///
    /// # Safety
    /// Same requirements and bounds argument as [`l2_squared`].
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm512_setzero_ps(); 2];
        let mut i = 0;
        while i + 32 <= n {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let offset = i + lane * 16;
                *acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(pa.add(offset)),
                    _mm512_loadu_ps(pb.add(offset)),
                    *acc,
                );
            }
            i += 32;
        }
        while i + 16 <= n {
            acc[0] = _mm512_fmadd_ps(
                _mm512_loadu_ps(pa.add(i)),
                _mm512_loadu_ps(pb.add(i)),
                acc[0],
            );
            i += 16;
        }
        let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(acc[0], acc[1]));
        while i < n {
            sum += *pa.add(i) * *pb.add(i);
            i += 1;
        }
        sum
    }

    /// Whether this CPU supports AVX-512F.
    #[inline(always)]
    pub(crate) fn available() -> bool {
        is_x86_feature_detected!("avx512f")
    }
}

// ─── Dispatch ────────────────────────────────────────────────────────────────
//
// Each entry point is defined once per configuration rather than branching
// inside one body, so no build ends up with an unreachable fallback arm.

/// Generates the three-way dispatch for one kernel.
macro_rules! dispatch {
    ($name:ident, $portable:ident) => {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        #[inline(always)]
        pub(crate) fn $name(a: &[f32], b: &[f32]) -> f32 {
            neon::$name(a, b)
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        #[inline(always)]
        pub(crate) fn $name(a: &[f32], b: &[f32]) -> f32 {
            // Widest available wins. Each check is a cached relaxed load.
            #[cfg(feature = "avx512")]
            {
                if avx512::available() {
                    // SAFETY: guarded by the runtime feature check.
                    return unsafe { avx512::$name(a, b) };
                }
            }
            if avx2::available() {
                // SAFETY: guarded by the runtime feature check.
                unsafe { avx2::$name(a, b) }
            } else {
                $portable(a, b)
            }
        }

        #[cfg(not(all(
            feature = "simd",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )))]
        #[inline(always)]
        pub(crate) fn $name(a: &[f32], b: &[f32]) -> f32 {
            $portable(a, b)
        }
    };
}

dispatch!(l2_squared, l2_squared_portable);
dispatch!(l1, l1_portable);
dispatch!(dot, dot_portable);

/// Squared L2 norm of a single vector.
#[inline(always)]
pub(crate) fn norm_squared(v: &[f32]) -> f32 {
    // A norm is a vector dotted with itself; reuse the tuned kernel.
    dot(v, v)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize, offset: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i + offset) as f32 * 0.37).sin() * 3.0 + 0.5)
            .collect()
    }

    /// Every dispatched kernel must agree with its scalar reference. The
    /// lengths cover below one vector block, exact multiples of every block
    /// width used (4, 8, 16), and multiples plus a scalar tail — the boundaries
    /// where a hand-written intrinsic loop is most likely to be wrong.
    #[test]
    fn vector_kernels_match_their_portable_reference() {
        for len in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 20, 23, 24, 31, 32, 33, 63, 64, 65,
            127, 128, 129, 384, 768, 769,
        ] {
            let a = sample(len, 0);
            let b = sample(len, 11);
            // Relative, not absolute: a fused multiply-add rounds once where
            // the portable path rounds twice, so the two legitimately differ
            // in the last ulp of a result whose magnitude grows with `len`.
            let close = |fast: f32, reference: f32| {
                (fast - reference).abs() <= 1e-5 * reference.abs().max(1.0)
            };

            let (fast, reference) = (l2_squared(&a, &b), l2_squared_portable(&a, &b));
            assert!(close(fast, reference), "l2_squared at len {len}: {fast} vs {reference}");
            let (fast, reference) = (l1(&a, &b), l1_portable(&a, &b));
            assert!(close(fast, reference), "l1 at len {len}: {fast} vs {reference}");
            let (fast, reference) = (dot(&a, &b), dot_portable(&a, &b));
            assert!(close(fast, reference), "dot at len {len}: {fast} vs {reference}");
            let (fast, reference) = (norm_squared(&a), dot_portable(&a, &a));
            assert!(close(fast, reference), "norm_squared at len {len}: {fast} vs {reference}");
        }
    }

    /// AArch64 graph construction must not depend on whether the default SIMD
    /// feature is enabled. HNSW is comparison-driven, so even a last-bit
    /// difference from FMA contraction or horizontal reduction can select a
    /// different graph and materially change build memory and query latency.
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    #[test]
    fn neon_kernels_are_bit_stable_with_portable_folds() {
        for len in [
            0usize, 1, 3, 4, 7, 8, 9, 15, 16, 31, 32, 63, 64, 100, 127, 128, 129, 384, 768,
            769,
        ] {
            for offset in [0usize, 1, 11, 97] {
                let a = sample(len, offset);
                let b = sample(len, offset + 23);
                assert_eq!(
                    l2_squared(&a, &b).to_bits(),
                    l2_squared_portable(&a, &b).to_bits(),
                    "l2_squared at len {len}, offset {offset}"
                );
                assert_eq!(
                    l1(&a, &b).to_bits(),
                    l1_portable(&a, &b).to_bits(),
                    "l1 at len {len}, offset {offset}"
                );
                assert_eq!(
                    dot(&a, &b).to_bits(),
                    dot_portable(&a, &b).to_bits(),
                    "dot at len {len}, offset {offset}"
                );
            }
        }
    }

    /// Unequal lengths must still fold over the common prefix, including when
    /// that prefix ends mid-block.
    #[test]
    fn unequal_lengths_use_the_common_prefix() {
        for (left, right) in [
            (6usize, 5usize),
            (20, 9),
            (9, 20),
            (17, 8),
            (8, 17),
            (33, 16),
            (100, 67),
        ] {
            let a = sample(left, 2);
            let b = sample(right, 23);
            let close = |fast: f32, reference: f32| {
                (fast - reference).abs() <= 1e-5 * reference.abs().max(1.0)
            };

            assert!(
                close(l2_squared(&a, &b), l2_squared_portable(&a, &b)),
                "l2_squared mismatch for {left}/{right}"
            );
            assert!(
                close(l1(&a, &b), l1_portable(&a, &b)),
                "l1 mismatch for {left}/{right}"
            );
            assert!(
                close(dot(&a, &b), dot_portable(&a, &b)),
                "dot mismatch for {left}/{right}"
            );
        }
    }

    /// Identical inputs must give exactly zero, with no accumulated fuzz from
    /// the horizontal reduction.
    #[test]
    fn identical_vectors_are_exactly_zero() {
        for len in [1usize, 8, 16, 17, 128, 769] {
            let a = sample(len, 5);
            assert_eq!(l2_squared(&a, &a), 0.0, "l2_squared at len {len}");
            assert_eq!(l1(&a, &a), 0.0, "l1 at len {len}");
        }
    }

    /// Kernels must not read past the end of a slice. Run under Miri or ASan
    /// this would catch an off-by-one in a loop bound; here it at least
    /// exercises every tail length against a tightly-sized allocation.
    #[test]
    fn kernels_stay_within_bounds_for_every_tail_length() {
        for len in 0..40usize {
            let a: Vec<f32> = sample(len, 1);
            let b: Vec<f32> = sample(len, 2);
            // Exact-capacity allocations so an over-read is more likely to
            // land outside the allocation.
            assert_eq!(a.len(), a.capacity().min(a.len()));
            let _ = l2_squared(&a, &b);
            let _ = l1(&a, &b);
            let _ = dot(&a, &b);
            let _ = norm_squared(&a);
        }
    }
}
