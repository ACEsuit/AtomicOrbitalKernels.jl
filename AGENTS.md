# AGENTS.md — AtomicOrbitalKernels.jl for coding agents

Architecture/intent/usage reference for agents working in this repo. For *how to
work* here (style, commit/PR rules, releasing) see `CLAUDE.md` — this file is
about *what the package is*. When code and this file disagree, the code wins;
update this file in the same PR.

## What this package is

`AtomicOrbitalKernels` provides fast, backend-agnostic
([KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl); CPU /
CUDA / Metal / ROCm) kernels for **atomic orbitals**, with two complementary
halves:

1. **Orbital evaluation** — batched `ϕ_{nlm}(𝐫, Z) = R_{nl}(r, Z)·Y_{lm}(𝐫̂)` for
   Gaussian- or Slater-type radials × real spherical harmonics, with **learnable**
   radial parameters `(ζ, D)`, value+gradient, Lux `(ps, st)`, and full
   `ChainRulesCore` differentiability in **positions and parameters**.
2. **Overlap integrals** — batched 2-center (and 3-center) Cartesian-Gaussian
   overlaps, interoperating with
   [GaussianBasis.jl](https://github.com/FermiQC/GaussianBasis.jl).

Intent: a GPU-ready, differentiable inner loop for research on learned atomic-
orbital bases (fit `(ζ, D)` from evaluation *and* overlap losses). It is **not** a
general quantum-chemistry integral library: only what's listed below is
implemented, and there are **no CPU fallbacks** to GaussianBasis.jl for the rest.

Built on Polynomials4ML (`AbstractP4MLBasis` interface, Lux + AD plumbing) and
SpheriCart (`SolidHarmonics`). Early/experimental but usable on CPU and GPU.

## Public API (exports)

- Orbital constructors: `gaussian_orbitals`, `slater_orbitals` (toy/example and
  `BasisSet`-loaded forms; see `src/orbitals/gaussianbasis.jl`, `utils.jl`).
- Evaluation: `evaluate`, `evaluate_ed` (value + spatial gradient).
- Overlap: `compile_basis`, `adapt_basis`, `batch_overlap[!]`,
  `batch_overlap_3c[!]`.

Not exported but commonly used internally: `pullback_ps`, `rrule` (param/position
gradients), `AtomicOrbitalKernels.Reference` (scalar oracle submodule).

## Two input/parameter worlds — do not mix them up

There are **two overlap paths** and **two input conventions**. Picking the wrong
one is the most common mistake here.

| Path | Build from | Input geometry | Differentiable? | Units |
|------|-----------|----------------|-----------------|-------|
| **GB / geometry-in** | `compile_basis(::BasisSet)` → `CompiledBasis` | `AbstractMatrix{<:Unitful.Length}` `(3,B)` | no | Unitful (any length; → Bohr at host) |
| **Orbital / learnable** | `compile_basis(::GaussianTypeOrbitals)` → `CartesianGTOBasis`, or call on the orbital directly | `Vector{<:PState}` (or `SVector{3}`) | **yes** (w.r.t. `(ζ,D)`) | scaled by the basis `lengthscale` |

- **`PState`** (DecoratedParticles): `x.𝐫` = `SVector{3}` position, `x.S` = species
  label. Orbital `evaluate` and the orbital-path `batch_overlap` take
  `Vector{PState}` (or a plain `Vector{SVector{3}}`, which defaults to species 1).
- The GB-path overlap kernels require **Unitful** position matrices and reject
  bare numeric matrices on purpose.

## Architecture & layering

### Orbital evaluation (`src/orbitals/`)
- `AtomicOrbitals{TR,LEN,TY,T} <: AbstractP4MLBasis` — product basis `Rnl·Ylm`.
  Aliases `GaussianTypeOrbitals`, `SlaterTypeOrbitals`. Field `lengthscale::T`.
- Radials `GaussianTypeRadials` / `SlaterTypeRadials <: GSRadials <:
  AbstractP4MLBasis`. Params `ζ, D` shaped **`[nRad × K × NZ]`** (`K` =
  contraction length, `NZ` = #species). Stored as `MArray` in the struct;
  returned as plain `Array` in `ps` (GPU-/AD-friendly).
- Three call layers (mirror these when adding methods): static
  `evaluate(b, X)` → Lux `evaluate(b, X, ps, st)` → kernel-facing
  `evaluate(b, Rs, sidx, ps, st)`. Same for `evaluate_ed`, `pullback_ps`.
- `ps = (Rnl = (ζ, D), Ylm = (;))`; `st = (Rnl = (poly,), Ylm = (Flm,), iR, iY)`
  (index maps + harmonics normalization live in **state**, so they move to the
  device). `evaluate_ref` is a forward-only oracle.
- Lux: `AtomicOrbitals` is already an `AbstractLuxLayer` (via P4ML).
  `LuxCore.initialparameters/initialstates/setup` and `orb(X, ps, st)` work; the
  package overloads `_init_luxparams`/`_init_luxstate`. LuxCore is **not** a direct
  dep (rides on P4ML; test-only in `Project.toml`).
- AD: `ChainRulesCore.rrule(evaluate, b, X, ps, st)` returns position **and**
  parameter cotangents; `pullback_ps` is a KA kernel (segmented species
  reduction). Runs on GPU.

### Overlap integrals (`src/integrals/`)
- McMurchie–Davidson E-coefficient recursion; one work-item per
  `(b, shell_a, shell_b[, shell_c])`; per-work-item `MArray` scratch sized by
  `Val{Lmax}`.
- GB path: `compiled_basis.jl` (`CompiledBasis`), `kernels_2c.jl` /
  `kernels_3c.jl`, `adapt.jl` (`adapt_basis` for GPU residency), `units.jl`
  (Unitful → Bohr).
- Orbital path: `orbital/cartesian_gto_basis.jl` (`CartesianGTOBasis`,
  species-aware: shared shell structure + per-species `ζ` + derived `coef`;
  **`D` is not stored**), `orbital/overlap_2c.jl` (forward kernel + `PState` API),
  `orbital/overlap_2c_grad.jl` (Lux `(ps,st)` value layer + analytic pullback
  kernel + `rrule`, **params only**; positions are `NoTangent`).
- `Reference` submodule (`integrals/reference/`) — non-exported scalar oracle the
  tests check against.

## Differentiable overlap (added in 0.0.4) — the model to follow

`batch_overlap(orb::GaussianTypeOrbitals, XA, XB, ps, st)` recomputes `coef` from
`ps.Rnl = (ζ, D)` each call (the *same* `ps` that drives `evaluate` — single
source of truth, transfers learned params straight back to the orbital). The
`rrule` returns `∂(ζ, D)` via: analytic kernel pullback (`∂coef` bilinear; `∂ζ`
from the raised-angular-momentum identity, E-tables run to `Lmax+2`) →
`_compile_coef` normalization VJP. Checked against ForwardDiff to ~1e-15.

## Conventions & gotchas (these bite)

- **`length_unit` is a REQUIRED kwarg** on every orbital constructor (no default —
  `UndefKeywordError` if omitted). Params stay in Bohr; a per-basis scalar
  `lengthscale` scales input positions → Bohr at the host boundary (kernels see
  Bohr). `lengthscale` is typed to the param float type.
- **Normalization `coef = D/√normsq` is scale-invariant in `D`.** Consequences:
  the learnable set is `(ζ, D)` *not* `(ζ, coef)` (coef can't recover D); and for
  **K=1 (uncontracted) bases `∂(overlap)/∂D ≡ 0`** — use `K ≥ 2` when testing
  D-gradients.
- **Padded primitives are `ζ=1, D=0` (inert).** Forward kernels skip `coef==0`
  pairs (value/NaN-safety); the overlap **pullback** kernel does **not** skip
  them (the true VJP/ForwardDiff include their derivative). Padded-slot gradients
  are not masked — same convention as the radial pullback.
- **Output is Cartesian Gaussian**, build `BasisSet(...; spherical=false)`. For
  `l ≥ 2` the Cartesian count `(l+1)(l+2)/2` exceeds the spherical `2l+1` (the
  extra functions are lower-`l` contamination — an open item to revisit).
- **`l ≥ 2` stride-aliasing wart**: contraction writes blocks with stride `N1 =
  2l+1` but reads back with stride `nbf`, scrambling d/f blocks. Preserved on
  purpose to match the equally-aliased `Reference`; only an independent
  quadrature oracle would catch it. New overlap code must reproduce it exactly
  (the diff pullback mirrors the forward stride/readback).
- **KA-only in shared paths** — no CUDA.jl-specific code (see `CLAUDE.md`).
  Backend is inferred from the array type.
- Do **not** hard-code `Float64`; compute in the promoted/working float type
  (so `ForwardDiff.Dual` and `Float32`/GPU flow). Differentiable value layers
  must set `FT = promote_type(positions, ζ, D)`.

## Build / test / run

```
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

- **GPU tests**: backend auto-detected (`TEST_BACKEND=CPU|CUDA|Metal|…` to force);
  no functional GPU → falls back to the CPU backend. `Pkg.test` forces
  `--check-bounds=yes`, which **breaks GPU kernel compilation**, so the GPU
  overlap testset is skipped under it. To exercise the GPU path:

  ```
  using Pkg; Pkg.test("AtomicOrbitalKernels"; julia_args=`--check-bounds=auto`)
  ```
- Param-gradient tests cross-check the analytic `pullback_ps`/`rrule` against
  ForwardDiff directional derivatives (the `lossζ`/`lossD` pattern). `ForwardDiff`
  and `LuxCore` are **test-only** deps; source AD uses only `ChainRulesCore`.
- Reference oracle: `AtomicOrbitalKernels.Reference.batch_S_pair_ref!` (and the
  3C `…batch_V_triple_ref!`).

## Scope: implemented vs not

Implemented: orbital evaluation (value + ∇, params + positions diff), 2C overlap
(GB + orbital/differentiable paths), 3C overlap (GB path only). **Not**
implemented: spherical-basis output, kinetic / nuclear-attraction / multipoles /
ERIs, 3C on the orbital path, Cartesian→spherical (`c2s`). Add integrals only as
needed; no GaussianBasis.jl fallbacks.
