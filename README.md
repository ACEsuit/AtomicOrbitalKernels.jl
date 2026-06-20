# AtomicOrbitalKernels

[![Build Status](https://github.com/ACEsuit/AtomicOrbitalKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ACEsuit/AtomicOrbitalKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)

Fast, backend-agnostic (CPU / CUDA / Metal / ROCm) kernels for **atomic
orbitals**, written with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) so a
single kernel set runs on every backend. The package has two complementary
halves:

1. **Atomic-orbital evaluation** — batched evaluation of orbital bases
   `ϕ_{nlm}(𝐫) = R_{nl}(r) · Y_{lm}(𝐫̂)` (Gaussian-type, Slater-type, or any
   radial × real spherical harmonics) with **learnable** radial parameters
   `(ζ, D)`. Values and spatial gradients (`evaluate` / `evaluate_ed`),
   `Lux`-compatible parameters/state, and full `ChainRulesCore`
   differentiability in **both positions and parameters**.
2. **Overlap integrals** — batched 2-center and 3-center Cartesian-Gaussian
   overlap integrals on top of
   [GaussianBasis.jl](https://github.com/FermiQC/GaussianBasis.jl).

Built on [Polynomials4ML.jl](https://github.com/ACEsuit/Polynomials4ML.jl),
[SpheriCart.jl](https://github.com/ACEsuit/SpheriCart.jl), and
[ACEbase.jl](https://github.com/ACEsuit/ACEbase.jl). This is early, somewhat
experimental software written with specific research projects in mind, but it is
already usable on CPU and GPU.

## Atomic-orbital evaluation

```julia
using AtomicOrbitalKernels, StaticArrays

basis = gaussian_orbitals()              # example Gaussian-type orbital basis
X = [ @SVector randn(3) for _ = 1:1000 ]

P     = evaluate(basis, X)               # values         (nX × nOrb)
P, dP = evaluate_ed(basis, X)            # values + ∇ϕ    (dP[i,k]::SVector{3})
```

`slater_orbitals(; K)` builds a Slater-type (optionally `K`-contracted) basis.
The radial parameters `(ζ, D)` are learnable and the basis is `Lux`-compatible:

```julia
using LuxCore, Random
rng = Random.default_rng()
ps = LuxCore.initialparameters(rng, basis)
st = LuxCore.initialstates(rng, basis)
P, _ = basis(X, ps, st)
```

Evaluation runs through KernelAbstractions, so the same calls execute on the GPU
when `X` and the parameters/state are device arrays. Gradients with respect to
positions **and** parameters are available through `ChainRulesCore` (e.g. via
Zygote); the parameter pullback (`pullback_ps`) and the `rrule` for `evaluate`
run on the GPU too.

## Overlap integrals

Batched, backend-agnostic Cartesian-Gaussian overlap integrals on top of
GaussianBasis.jl.

| Category                          | Status in `AtomicOrbitalKernels.jl`                                  |
| --------------------------------- | -------------------------------------------------------------------- |
| 2-center Cartesian overlap        | **Implemented** (mirrors `GaussianBasis.overlap`)                    |
| 3-center Cartesian overlap        | **Added** (extension: not present in GaussianBasis.jl)               |
| Spherical-basis output            | Not implemented                                                      |
| Kinetic, nuclear attraction       | Not implemented                                                      |
| Multipoles (dipole, quadrupole …) | Not implemented                                                      |
| ERIs (2e2c, 2e3c, 2e4c)           | Not implemented                                                      |
| Gradients                         | Not implemented                                                      |

Integrals beyond 2C and 3C overlap will be added as needed. The package provides
**no fallbacks** to GaussianBasis.jl for unimplemented operations — if you need
them on CPU, call GaussianBasis.jl directly.

The 3-center overlap `V_{μνλ}(b) = ∫ φ_μ(r - A_b) φ_ν(r - B_b) φ_λ(r - C_b) dr`
has no equivalent in GaussianBasis.jl. Its `ERI_2e3c` is a different,
2-electron, 3-center integral.

```julia
using GaussianBasis, Molecules, StaticArrays, Unitful
using AtomicOrbitalKernels

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

### Conventions

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
  reference (`AtomicOrbitalKernels.Reference`).

### Reference implementation

The pedagogical scalar implementation lives in the non-exported submodule
`AtomicOrbitalKernels.Reference`. It is what the test suite checks against and
is available to downstream code:

```julia
using AtomicOrbitalKernels: Reference
# Reference.batch_S_pair_ref!(out, BS, posA, posB)   # positions: plain Å
# Reference.batch_V_triple_ref!(out, BS, posA, posB, posC)
```

The reference exposes the underlying McMurchie–Davidson E-coefficient
recursion (`generate_E_matrix!`, `generate_E3_matrix!`) and per-shell-pair /
shell-triple writes (`generate_S_pair!`, `generate_V_triple!`).

## Benchmarks & scaling

A `PkgBenchmark` suite for the orbital evaluation and pullback kernels lives in
[benchmark/](benchmark) (run with `using PkgBenchmark; benchmarkpkg(...)`). A
separate scaling-sweep driver for the overlap kernels lives at
[scaling/scaling.jl](scaling/scaling.jl) with its own `scaling/Project.toml`; it
runs the 2C and 3C kernels at batch sizes `B = 2^7 … 2^14` for a single backend
per invocation and prints a markdown timing table.

```
# CPU using all physical cores
julia --project=scaling -t auto scaling/scaling.jl

# GPU (Float32) — pass the backend package name (must be installed)
julia --project=scaling scaling/scaling.jl CUDA
julia --project=scaling scaling/scaling.jl Metal
```

The KA `CPU` backend parallelises across Julia threads, so the CPU numbers
depend on `--threads` / `-t` / `JULIA_NUM_THREADS`; thread count is irrelevant
once a GPU backend is selected.

## Testing

```
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

GPU tests are opt-in: the backend is auto-detected (set `TEST_BACKEND` to force
`CPU`/`CUDA`/`Metal`/…), and on a machine with no functional GPU the device
tests fall back to the CPU backend.
