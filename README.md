# AtomicOrbitalKernels

[![Build Status](https://github.com/ACEsuit/AtomicOrbitalKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ACEsuit/AtomicOrbitalKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)

Fast, GPU-ready (CPU / CUDA / Metal / ROCm, via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl))
kernels for **atomic-orbital evaluation** and **Cartesian-Gaussian overlap
integrals**, with **learnable, differentiable** radial parameters `(ζ, D)`:

- **Evaluation** — batched `ϕ_{nlm}(𝐫, Z) = R_{nl}(r, Z) · Y_{lm}(𝐫̂)` for
  Gaussian- or Slater-type radials × real spherical harmonics, values and spatial
  gradients, `Lux`-style parameters/state, and full `ChainRulesCore`
  differentiability in **positions and parameters**.
- **Overlap integrals** — batched 2-center (and 3-center) Cartesian-Gaussian
  overlaps; the 2-center orbital path is **differentiable w.r.t. `(ζ, D)`**, so a
  basis can be trained directly from an overlap loss.

Built on [Polynomials4ML.jl](https://github.com/ACEsuit/Polynomials4ML.jl) and
[SpheriCart.jl](https://github.com/ACEsuit/SpheriCart.jl). It interoperates with
[GaussianBasis.jl](https://github.com/FermiQC/GaussianBasis.jl) — used to load
standard basis-set parameters, and available as an alternate "geometry-in"
overlap path. This is early, research-oriented software, but already usable on
CPU and GPU.

## Installation

```julia
using Pkg; Pkg.add("AtomicOrbitalKernels")
```

## Quick start

Build a **multi-element** basis from a standard basis set and evaluate it. Each
input point is a `DecoratedParticles.PState` carrying a position (`𝐫`) and the
species (`S`) of the atom it is centred on:

```julia
using AtomicOrbitalKernels, StaticArrays
using AtomsBase: ChemicalSpecies
using DecoratedParticles: PState

# cc-pVDZ for C, N and O; coordinates will be given in Ångström (see "Units")
basis = gaussian_orbitals("cc-pvdz", [:C, :N, :O]; length_unit = :angstrom)

# evaluate at 1000 points, each tagged with its atom's species
X = [ PState(𝐫 = @SVector(randn(3)), S = ChemicalSpecies(:C)) for _ = 1:1000 ]

P     = evaluate(basis, X)        # (nX × nOrb) matrix,  P[i,k] = ϕ_k(X[i])
P, dP = evaluate_ed(basis, X)     # values + spatial gradients  (dP[i,k]::VState)
```

Other constructors: `gaussian_orbitals(BasisSet(...))` from an existing
GaussianBasis set, `gaussian_orbitals([:C => "cc-pvdz", :H => "sto-3g"]; …)` for a
per-element mix, and `gaussian_orbitals(; length_unit)` / `slater_orbitals(; K,
length_unit)` for small example bases. A basis with a single species also accepts
plain `SVector{3}` coordinates (taken as species 1) instead of `PState`s.

Evaluation runs through KernelAbstractions, so the same calls execute on the GPU
when the inputs, parameters and state are device arrays.

## Units

`length_unit` is a **required** keyword at construction — there is no default, so
the unit of the coordinates you evaluate at is always explicit (a stray Å-vs-Bohr
mix-up is otherwise silent). Accepted values: `:bohr`, `:angstrom` / `:Å`, or any
Unitful length (e.g. `u"nm"`).

The radial parameters `(ζ, D)` are stored in **atomic units (Bohr)**. Input
positions are interpreted in the basis's `length_unit` and scaled to Bohr at the
host boundary (a per-basis `lengthscale` factor); the kernels only ever see Bohr.
So a basis built with `length_unit = :angstrom` evaluated at `X` gives the same
values as a `:bohr` basis evaluated at `ang2bohr .* X`.

The GaussianBasis geometry-in overlap path (below) uses a different, equally
explicit convention: positions are passed as `Unitful.Length` matrices and
stripped to Bohr — see that section.

## Learnable parameters & differentiation

The radial `(ζ, D)` are trainable, and the basis follows the `Lux` layer
interface. `LuxCore.setup` returns `ps = (Rnl = (ζ, D), Ylm = (;))` and the
non-trainable state `st`:

```julia
using LuxCore, Zygote, Random
rng = Random.default_rng()

ps, st = LuxCore.setup(rng, basis)     # ps.Rnl.ζ, ps.Rnl.D are the trainable params
P, _   = basis(X, ps, st)              # Lux forward  (== evaluate(basis, X, ps, st))

# evaluation is differentiable in positions AND parameters (ChainRules ⇒ Zygote):
∂ps = Zygote.gradient(p -> sum(abs2, evaluate(basis, X, p, st)), ps)[1]
```

The 2-center overlap on the orbital basis is likewise differentiable w.r.t.
`(ζ, D)` (the same `ps`), so a basis can be fit directly from an overlap loss —
learned parameters transfer straight back to the orbitals:

```julia
sp = ChemicalSpecies(:C)
XA = [ PState(𝐫 = 0.4 .* @SVector(randn(3)),               S = sp) for _ = 1:16 ]
XB = [ PState(𝐫 = 0.4 .* @SVector(randn(3)) .+ SA[1.2,0,0], S = sp) for _ = 1:16 ]

S   = batch_overlap(basis, XA, XB, ps, st)                 # (nbf × nbf × B)
∂ps = Zygote.gradient(p -> sum(abs2, batch_overlap(basis, XA, XB, p, st)), ps)[1]
```

The parameter pullbacks and `rrule`s run on the GPU too. (Overlap gradients w.r.t.
atom positions are not implemented yet — only w.r.t. the parameters.)

## Overlap integrals

Batched, backend-agnostic Cartesian-Gaussian overlap integrals
`S_{μν}(b) = ⟨φ_μ(𝐫 − A_b) | φ_ν(𝐫 − B_b)⟩`. There are two entry points:

**Orbital-native (differentiable).** Compile an `AtomicOrbitals` basis once and
call it with `PState` inputs (the same species-tagged points as `evaluate`). This
is the path used above and the one that supports parameter gradients:

```julia
cob = compile_basis(basis)             # -> species-aware CartesianGTOBasis
S   = batch_overlap(cob, XA, XB)       # inference; use the ps/st form to differentiate
```

**GaussianBasis geometry-in.** For interop / pure inference from an existing
`BasisSet`, compile it directly and pass **Unitful** `(3, B)` position matrices
(any length unit; stripped to Bohr). Build the set with `spherical = false`:

```julia
using GaussianBasis, StaticArrays, Unitful

BS    = BasisSet("def2-svp", "Si 0.0 0.0 0.0"; spherical = false)
basis = compile_basis(BS)
N, B  = basis.nbf_total, 1024

posA = randn(3, B) .* 0.5 .* u"angstrom"
posB = (randn(3, B) .* 0.5 .+ SA[1.5, 0.0, 0.0]) .* u"angstrom"
out  = zeros(Float64, N, N, B)
batch_overlap!(out, basis, posA, posB)

# GPU: move the basis (adapt_basis(basis, CuArray, Float32)) and preallocate `out`
# on the device, then call batch_overlap! with the same Unitful positions.
```

The 3-center overlap
`V_{μνλ}(b) = ∫ φ_μ(𝐫 − A_b) φ_ν(𝐫 − B_b) φ_λ(𝐫 − C_b) d𝐫` (no equivalent in
GaussianBasis.jl) uses the same geometry-in basis with one more position matrix;
its output scales as `O(N³·B)`, so use a smaller `B`:

```julia
posC = (randn(3, 128) .* 0.5 .+ SA[0.7, 1.2, 0.3]) .* u"angstrom"
out3 = zeros(Float64, N, N, N, 128)
batch_overlap_3c!(out3, basis, posA[:, 1:128], posB[:, 1:128], posC)
```

### Status

| Feature                                    | Status                                            |
| ------------------------------------------ | ------------------------------------------------- |
| 2-center overlap (Cartesian)               | **Implemented** — orbital-native + geometry-in    |
| 2-center overlap parameter gradients `(ζ,D)` | **Implemented** — orbital-native path            |
| 3-center overlap (Cartesian)               | **Implemented** — geometry-in path                |
| Overlap position gradients                 | Not implemented                                   |
| Spherical-basis output                     | Not implemented (Cartesian only)                  |
| Kinetic, nuclear attraction, multipoles, ERIs | Not implemented                                |

There are **no fallbacks** to GaussianBasis.jl for unimplemented operations — call
it directly on CPU if you need them.

### Conventions

- **Cartesian output.** Overlaps are over Cartesian Gaussian shells; build any
  `BasisSet` with `spherical = false`.
- **Basis-function ordering / `l ≥ 2` aliasing.** The contraction uses a linear
  stride of `2l+1` rather than the Cartesian `nbf = (l+1)(l+2)/2`, which produces
  deliberate index aliasing for `l ≥ 2` — preserved bit-for-bit against the
  bundled scalar reference (`AtomicOrbitalKernels.Reference`). Relatedly, for
  `l ≥ 2` a Cartesian shell has more functions than the `2l+1` spherical ones (the
  extras are lower-`l` contamination); this is a known item to revisit.

### Reference implementation

The pedagogical scalar implementation lives in the non-exported submodule
`AtomicOrbitalKernels.Reference` (what the test suite checks against). It exposes
the McMurchie–Davidson E-coefficient recursion and per-shell-pair / shell-triple
writes:

```julia
using AtomicOrbitalKernels: Reference
# Reference.batch_S_pair_ref!(out, BS, posA, posB)      # positions: plain Å matrices
# Reference.batch_V_triple_ref!(out, BS, posA, posB, posC)
```

## Benchmarks & scaling

A `PkgBenchmark` suite for the orbital evaluation and pullback kernels lives in
[benchmark/](benchmark). A separate scaling-sweep driver for the overlap kernels
is at [scaling/scaling.jl](scaling/scaling.jl) (with its own `scaling/Project.toml`);
it runs the 2C and 3C kernels at batch sizes `B = 2^7 … 2^14` for one backend per
invocation and prints a markdown timing table.

```
julia --project=scaling -t auto scaling/scaling.jl     # CPU, all physical cores
julia --project=scaling scaling/scaling.jl CUDA        # GPU (Float32); or Metal
```

The KA `CPU` backend parallelises across Julia threads, so CPU numbers depend on
`--threads` / `-t` / `JULIA_NUM_THREADS`; thread count is irrelevant on a GPU.

## Testing

```
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

GPU tests are opt-in: the backend is auto-detected (set `TEST_BACKEND` to force
`CPU` / `CUDA` / `Metal` / …), and on a machine with no functional GPU the device
tests fall back to the CPU backend. `Pkg.test` forces `--check-bounds=yes`, which
GPU kernels can't compile under, so the GPU overlap testset is skipped under it.
To exercise the GPU path:

```julia
using Pkg; Pkg.test("AtomicOrbitalKernels"; julia_args=`--check-bounds=auto`)
```
