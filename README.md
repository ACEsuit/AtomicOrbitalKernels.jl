# GaussianBasisKernels

[![Build Status](https://github.com/cortner/GaussianBasisKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cortner/GaussianBasisKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)

Batched, backend-agnostic GPU/CPU kernels for Cartesian-Gaussian basis-set
integrals on top of [GaussianBasis.jl](https://github.com/FermiQC/GaussianBasis.jl),
built with [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
so the same code runs on CPU, CUDA, Metal, and (untested) ROCm.

## Scope

| Category                          | Status in `GaussianBasisKernels.jl`                                  |
| --------------------------------- | -------------------------------------------------------------------- |
| 2-center Cartesian overlap        | **Implemented** (mirrors `GaussianBasis.overlap`)                    |
| 3-center Cartesian overlap        | **Added** (extension: not present in GaussianBasis.jl)               |
| Spherical-basis output            | Not implemented                                                      |
| Kinetic, nuclear attraction       | Not implemented                                                      |
| Multipoles (dipole, quadrupole …) | Not implemented                                                      |
| ERIs (2e2c, 2e3c, 2e4c)           | Not implemented                                                      |
| Gradients                         | Not implemented                                                      |

Integrals beyond 2C and 3C overlap will be added as we need them. The package
provides **no fallbacks** to GaussianBasis.jl for unimplemented operations —
if you need them on CPU, call GaussianBasis.jl directly.

The 3-center overlap `V_{μνλ}(b) = ∫ φ_μ(r - A_b) φ_ν(r - B_b) φ_λ(r - C_b) dr`
has no equivalent in GaussianBasis.jl. Its `ERI_2e3c` is a different,
2-electron, 3-center integral.

## Quickstart

```julia
using GaussianBasis, Molecules, StaticArrays, Unitful
using GaussianBasisKernels

# 1. Build a Cartesian basis set the usual GaussianBasis.jl way.
atom = Molecules.Atom(14, 28.0855, SA[0.0, 0.0, 0.0])
BS = BasisSet("def2-SVP", [atom]; spherical=false, lib=:acsint)

# 2. Compile to the type-stable, struct-of-arrays form once.
basis = compile_basis(BS)
N = basis.nbf_total

# 3. Batched 2-center overlap on the CPU. Positions MUST carry length units.
B = 1024
posA = randn(3, B) .* 0.5 .* u"angstrom"
posB = (randn(3, B) .* 0.5 .+ SA[1.5, 0.0, 0.0]) .* u"angstrom"

out = zeros(Float64, N, N, B)
batch_overlap!(out, basis, posA, posB)

# 4. Same call on GPU. Move the compiled basis and preallocate `out` on the device.
# using CUDA
# basis_gpu = adapt_basis(basis, CuArray, Float32)
# out_gpu   = CuArray(zeros(Float32, N, N, B))
# batch_overlap!(out_gpu, basis_gpu, posA, posB)   # positions still in Å (or any Unitful.Length)

# 3-center overlap follows the same pattern, with one more position matrix:
posC = (randn(3, 128) .* 0.5 .+ SA[0.7, 1.2, 0.3]) .* u"angstrom"
out3 = zeros(Float64, N, N, N, 128)
batch_overlap_3c!(out3, basis, posA[:, 1:128], posB[:, 1:128], posC)
```

## Conventions

- **Positions** must be passed as `AbstractMatrix{<:Unitful.Length}` (size
  `(3, B)`). Any length unit is accepted — `u"angstrom"`, `u"bohr"`, `u"nm"`,
  etc. — and is stripped + converted to Bohr at the host boundary before the
  kernel launches. Plain numeric matrices are deliberately rejected with a
  clear `ArgumentError`. This is intentionally stricter than the
  GaussianBasis.jl convention (which treats `atom.xyz` as bare Å).
- **Output basis** is **Cartesian** Gaussian. Build your `BasisSet` with
  `spherical=false`.
- **Basis-function ordering** mirrors the prototype's `generate_S_pair!`
  exactly. For shells with `l ≥ 2` the contraction uses a linear stride of
  `2l+1` rather than `nbf = (l+1)(l+2)/2`, which produces deliberate index
  aliasing — preserved for bit-for-bit compatibility with the bundled scalar
  reference (`GaussianBasisKernels.Reference`).
- **Element type** is determined by `eltype(out)`. Use `Float64` on CPU,
  `Float32` on GPU (Apple Metal requires it; CUDA tends to be faster with it).

## Reference implementation

The pedagogical scalar implementation lives in the non-exported submodule
`GaussianBasisKernels.Reference`. It is what the test suite checks against and
is available to downstream code:

```julia
using GaussianBasisKernels: Reference
# Reference.batch_S_pair_ref!(out, BS, posA, posB)   # positions: plain Å
# Reference.batch_V_triple_ref!(out, BS, posA, posB, posC)
```

The reference exposes the underlying McMurchie–Davidson E-coefficient
recursion (`generate_E_matrix!`, `generate_E3_matrix!`) and per-shell-pair /
shell-triple writes (`generate_S_pair!`, `generate_V_triple!`).

## Benchmarks

A benchmark driver lives at [bench/batch_bench.jl](bench/batch_bench.jl) with
its own `bench/Project.toml`. Run from the repo root:

```
julia --project=bench bench/batch_bench.jl
```

It validates both 2C and 3C against the scalar reference and prints CPU
(BenchmarkTools `@btime`) timings; GPU timings appear automatically if a
functional `CUDA` or `Metal` backend is available in the bench project.

## Testing

```
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

GPU tests (`test/test_gpu.jl`) are opt-in and skipped silently when neither
CUDA nor Metal is functional.
