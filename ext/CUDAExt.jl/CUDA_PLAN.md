# NVIDIA-specific optimisation plan for AtomicOrbitalKernels

This document is a stand-alone roadmap for adding a hand-tuned NVIDIA
(CUDA.jl) implementation of the 2C / 3C Cartesian-Gaussian overlap kernels
on top of the existing portable KernelAbstractions kernels. It was written
on a laptop without GPU access for later pickup on a machine with an A100
(or similar) NVIDIA GPU.

The KA kernels in [src/kernels_2c.jl](../../src/kernels_2c.jl) and
[src/kernels_3c.jl](../../src/kernels_3c.jl) stay as the portable fallback;
the CUDA path is added as a package extension `CUDAExt` that takes over
when `out isa CuArray`. Public API (`compile_basis`, `adapt_basis`,
`batch_overlap!`, `batch_overlap_3c!`) is preserved bit-for-bit.

## Where the current kernel stands

Both KA kernels follow the same pattern:

- **Granularity**: one CUDA thread per work-item. 2C launches with
  `ndrange = (B, ns, ns)`, 3C with `(B, ns, ns, ns)`.
- **Per-thread scratch** (sized at compile time by `Val{Lmax}`,
  `eltype = FT = Float32` on GPU):
  - 2C: `E` of shape `(2Lmax+2, Lmax+1, Lmax+1, 3)` plus `blk` of size
    `((Lmax+1)(Lmax+2)/2)²`. At `Lmax=2`: ≈ 800 B/thread.
  - 3C: `E3` of shape `(3Lmax+3, Lmax+1, Lmax+1, Lmax+1, 3)` plus
    `blk` of size `NBF³`. At `Lmax=2`: ≈ 3.8 KB/thread; at `Lmax=3`:
    ≈ 16 KB/thread.
- **Recursion**: branch-eliminated McMurchie–Davidson E-coefficient build,
  per axis, inside a per-primitive-pair (2C) or per-primitive-triple (3C)
  contraction loop. Loop bounds depend on the dynamic `l_a, l_b[, l_c]`
  of the current shell triple.
- **Writes**: `out[row_off + i, col_off + j, b]` for 2C (and
  `+, page_off + k` for 3C). Output is column-major, so dim 1 (`i`) is
  fastest-varying. Threads in the same warp typically share `(s_a, s_b)`
  and span consecutive `b`, which means **every thread's writes are to a
  different output slab** — classic non-coalesced access at warp
  granularity.
- **What KA gives up vs. raw CUDA.jl**: no native shared-memory hierarchy
  control, no warp shuffles, no cooperative groups, no PTX intrinsics, no
  per-shell-type kernel specialisation.

Reference Metal numbers (def2-SVP / Si, `Lmax=2`, `N=19`, Float32) from the
[scaling sweep](../../scaling/scaling.jl):

| B     |   2C    |   3C    |
| ----: | ------: | ------: |
|   128 | 0.64 ms |  4.5 ms |
|  1024 | 0.96 ms |  24 ms  |
| 16384 | 5.3 ms  | 427 ms  |

On NVIDIA the absolute numbers will be lower (more SMs, more cache) but the
shape — kernel-launch / occupancy-bound at small B, compute / local-memory
bound at large B, output-bandwidth contribution growing with B — is the
same.

## Optimisation axes and rough speedup ranges (vs. current KA kernel on the same NVIDIA GPU)

### Axis 1 — Per-(l_a, l_b[, l_c]) kernel specialisation  · **2–4×**

Currently the recursion body runs general `for i in 2:(l_a+1)` loops, so
the inner-loop trip count is data-dependent. A CUDA build can emit one
kernel per (l_a, l_b) (for 2C) or (l_a, l_b, l_c) (for 3C) shell-type
combination and have nvcc fully unroll every loop with static bounds.
Each shell pair / triple then dispatches to its specialised kernel. With
`Lmax ≤ 2` this is 9 specialisations (3×3) for 2C and 27 for 3C. nvcc can
then keep `E` / `E3` entirely in registers for the simpler types (s/s,
s/p) and the compiler schedules the FMAs aggressively.

**Why it's NVIDIA-specific**: KA can in principle do this via `Val{(la, lb)}`
plus a launch dispatcher, but the compile-time cost and the lack of a CUDA
PTX-level intrinsic for warp-aware code makes the payoff much smaller
without the rest of the NVIDIA-only optimisations below.

### Axis 2 — Coalesced output writes  · **2–4× at large B**

At `Lmax=2`, 2C: each thread writes `nbf_a · nbf_b ≤ 36` Float32s. With
the current layout, a warp's 32 threads all share `(s_a, s_b)` and differ
in `b`; their writes land in distant memory because `b` is the slowest
output dim. A CUDA implementation can swap the work assignment: one warp
handles one `(s_a, s_b, b)` and the 32 threads collaboratively compute
the `nbf_a × nbf_b` block, writing **one row per cycle, 32 elements at a
time**. For 3C it's even more dramatic because the per-thread block is
larger (`nbf³`). At `B = 16384` this is plausibly the single biggest win.

### Axis 3 — Shared-memory E-coefficient scratch + warp cooperation  · **1.5–3×**

For `Lmax ≤ 2`, the `E` / `E3` arrays probably stay in registers and this
axis is small. For `Lmax = 3` (≈ 5 KB `E3` per thread on 3C) we are deep
in **local-memory spill territory**, which is global-memory-backed on
NVIDIA and devastating to throughput. A CUDA build can put `E3` in shared
memory (48–100 KB/SM available; A100 has 192 KB configurable L1/shared)
shared by a thread block of 32–128 threads that collaboratively build the
recursion. The recursion has a sequential dependency in `t` and along
each (i, j, k) chain, so cooperation gains are moderate, but **avoiding
the spill alone** is large.

### Axis 4 — Tensor-core / TF32 contraction  · **1.5–4× for the contraction step, less overall**

The final contraction inside both kernels is a small dense product of
three E-coefficient vectors times a prefactor. If many shell-pairs at the
same `b` are batched into one `mma.sync.aligned` instruction, TF32 tensor
cores can deliver ≈ 8× the FP32 throughput of CUDA cores on Ampere+
(A100 included) / Hopper. The contraction is a minority of the total
runtime for low Lmax, so the wall-clock impact is modest, but it grows
with Lmax.

### Axis 5 — Persistent kernels / launch-overhead amortisation  · **1.2–2× at small B**

At B = 128 the kernel launch on Metal is ≈ 640 μs and the actual compute
is microseconds. NVIDIA suffers the same shape but with CUDA streams +
graphs and persistent kernels you can keep one launch live across many
problem instances. Only matters for the small-B regime; irrelevant for
B ≥ 4096.

### Axis 6 — Memory layout / AoS→SoA tweaks on the basis data  · **1.1–1.3×**

Minor. The `CompiledBasis` SoA in
[src/compiled_basis.jl](../../src/compiled_basis.jl) is already friendly.
There may be small wins from interleaving `(coef, α)` pairs so each thread
issues one 128-bit load instead of two 32-bit loads.

### Axis 7 — Auxiliary basis / sparse 3C  · **separate axis, not a speedup of the same work**

For real 3C use cases the auxiliary basis is small and most
(s_a, s_b, s_c) triples don't need full evaluation. A CUDA-specific path
can also add screening; that's a separate algorithmic win on top of the
per-kernel speedups above.

## Composite estimate

Multiplying the axes is not honest — they overlap (coalescing-aware
specialised kernels with shared-memory E3 is one design, not three). A
more realistic synthesis:

- **2C, `Lmax ≤ 2` (e.g. def2-SVP), large B**: 3–8× over the current KA
  kernel on the same NVIDIA GPU. The bulk comes from coalescing (Axis 2)
  and kernel specialisation (Axis 1). At small B the launch-overhead
  axis dominates and the win drops to 1.5–2×.

- **2C, `Lmax ≥ 3`**: 5–15×. Local-memory spillage (Axis 3) becomes a
  real problem in the portable kernel; the CUDA build can avoid it.

- **3C, `Lmax ≤ 2`, large B**: 5–15×. Bigger per-thread block (Axis 2
  pays more), bigger E3 scratch (Axis 3 pays more), bigger fraction of
  time in contraction (Axis 4 visible).

- **3C, `Lmax ≥ 3`**: 10–30×, dominated by avoiding spill on E3.

Anything beyond ≈ 30× over the current implementation would be surprising
without changing the *algorithm* (Obara–Saika vs. McMurchie–Davidson
choice, screening, asymptotic-far-region approximations, etc.).

For external calibration: published full-ERI GPU codes (e.g. LibintX,
BrianQC, GPU4PySCF) report 5–50× speedups over CPU on whole 4-center ERI
evaluations. Overlap is much cheaper per shell pair than ERI, so the GPU
wins are at the lower end of that range.

## Caveats — what this costs

- **Lines of code & maintenance**: A full CUDA.jl kernel with the
  specialisations, warp-cooperative scratch, and coalesced layout is
  roughly 3–5× the LoC of the current portable kernel, plus per-shell-type
  generated dispatch tables. Diverging from KA gives up Metal / AMDGPU /
  CPU coverage unless you keep the KA kernel as a fallback (doubles the
  surface).
- **The CPU benchmark is the moving target**: the same kind of work
  (kernel specialisation, AVX-512 with explicit masks, NUMA-aware
  partitioning) would also speed up the CPU side substantially. CPU vs.
  GPU comparisons assume neither side gets that treatment.
- **Realism on small problems**: at the basis sizes we're benchmarking
  (`N = 19`), the work per shell-pair is small enough that occupancy and
  launch overhead matter as much as compute optimisation. The estimates
  above assume B large enough to hide launch.
- **Float32 accuracy is the binding constraint sooner**: deeper recursion
  trees (`Lmax ≥ 4`) lose meaningful precision in Float32; some of the
  speedup is unspendable unless you go to TF32-with-FP32-accumulate or
  mixed precision.

## Staged implementation plan

When picked up on an A100-class machine:

1. **Wire up the extension skeleton.** Add `[extensions] CUDAExt = "CUDA"`
   and `[weakdeps] CUDA = "..."` to [Project.toml](../../Project.toml).
   Create `ext/CUDAExt.jl/CUDAExt.jl` that imports `CUDA` and
   `AtomicOrbitalKernels`, and defines `batch_overlap!(out::CuArray, ...)`
   / `batch_overlap_3c!(out::CuArray, ...)` methods that take over from
   the KA path. Initial body: just call into the same KA kernel — verify
   the extension loads and tests still pass.

2. **Axis 2 alone (coalesced output).** Re-target the launch geometry so
   that within a warp, threads cooperate on a single shell-pair block at
   one `b`. Re-run [scaling/scaling.jl](../../scaling/scaling.jl) — expect
   1.5–3× at B ≥ 1024. This is the single highest-leverage change and
   should land first.

3. **Axis 1 (per-shell-type specialisation).** Generate one CUDA kernel
   per `(l_a, l_b)` (and `(l_a, l_b, l_c)` for 3C) using either
   `@generated` or Julia metaprogramming. Dispatch from the host wrapper
   based on the shell triple. Expect another 1.5–2×, most visible on the
   d/d 2C and d/d/d 3C pairs.

4. **Axis 3 (shared-memory E / E3).** Move `E3` into `@cuStaticSharedMem`
   shared by a thread block. Most useful at higher Lmax — re-run scaling
   on `def2-TZVPP` (which adds f shells for Si) or another basis with
   `Lmax ≥ 3` to see the win clearly. On A100 the configurable 192 KB L1
   / shared partition makes this comfortable.

5. **Axis 5 (CUDA graphs)**, **Axis 4 (TF32 contractions)**, **Axis 6
   (layout tweaks)** — optional follow-ups, lower priority.

Each stage validated by:

- `julia --project -e 'using Pkg; Pkg.test()'` — correctness pinned to
  `AtomicOrbitalKernels.Reference` via the existing tests in
  [test/test_overlap_2c.jl](../../test/test_overlap_2c.jl) and
  [test/test_overlap_3c.jl](../../test/test_overlap_3c.jl). The CUDA
  branch goes through the same assertions, so any regression shows up.
- `julia --project=scaling scaling/scaling.jl CUDA` for an
  apples-to-apples wall-clock comparison against the KA-CUDA baseline.
  Recommended cadence: capture a baseline table before stage 1, then a
  new table after each stage; keep them in this directory as
  `scaling_baseline.md`, `scaling_after_axis2.md`, etc., so the
  speedup-per-stage is documented.

## A100-specific notes

- **Compute capability 8.0** (Ampere). Tensor cores support TF32, BF16,
  FP16, INT8. The TF32 path is the relevant one (matches `FT = Float32`
  externally; cores compute in 19-bit mantissa internally, accumulate in
  FP32). Enable with `CUDA.math_mode!(CUDA.FAST_MATH)` plus tensor-core
  intrinsics.
- **L1/shared partition**: 192 KB combined; selectable in 28/164,
  100/100, 132/64, etc. configurations. For Axis 3, request the
  shared-heavy partition.
- **Async copies** (`cp.async`): A100 has hardware-async global→shared
  copies, useful when streaming basis data into shared once and reusing
  across many shell pairs. Worth a try in stage 4.
- **Occupancy reality check**: with `~16 KB` of `E3` per warp under
  Axis 3, occupancy is bounded to roughly 4–8 warps per SM at `Lmax=3`.
  Plenty to hide memory latency but not a wide margin.

## Repository pointers for the next session

- Portable kernels: [src/kernels_2c.jl](../../src/kernels_2c.jl),
  [src/kernels_3c.jl](../../src/kernels_3c.jl).
- Compiled basis (SoA, GPU-shippable):
  [src/compiled_basis.jl](../../src/compiled_basis.jl).
- Device transfer helper: [src/adapt.jl](../../src/adapt.jl).
- Unit-conversion adapter (Bohr boundary):
  [src/units.jl](../../src/units.jl).
- Correctness oracle:
  [src/reference/](../../src/reference/) (a non-exported submodule).
- Test harness pinned to the oracle:
  [test/test_overlap_2c.jl](../../test/test_overlap_2c.jl),
  [test/test_overlap_3c.jl](../../test/test_overlap_3c.jl).
- Benchmark / scaling driver:
  [scaling/scaling.jl](../../scaling/scaling.jl).
