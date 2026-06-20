"""
    AtomicOrbitalKernels

Fast batched GPU kernels for Cartesian-Gaussian basis-set integrals, built on
KernelAbstractions. Currently implements:

- 2-center Cartesian-Gaussian overlap (mirrors `GaussianBasis.overlap`)
- 3-center Cartesian-Gaussian overlap (extension; not in `GaussianBasis.jl`)

Every other integral (spherical basis, kinetic, nuclear attraction, ERIs,
gradients, multipoles) is **not implemented** in this package. We will add
integrals as we need them; there are no GaussianBasis.jl fallbacks for
unimplemented operations — call GaussianBasis.jl directly for those on CPU.

See the README for a quickstart and a full scope comparison with
GaussianBasis.jl.
"""
module AtomicOrbitalKernels

using StaticArrays
using KernelAbstractions
const KA = KernelAbstractions
using Unitful

# Direct dependency: `compile_basis` consumes a `GaussianBasis.BasisSet`. The
# kernels themselves are GaussianBasis-free — if you ever need to ship a
# lightweight inference-only build, the obvious refactor is to move
# `compile_basis(::BasisSet)` into a `GaussianBasisExt` package extension.
using GaussianBasis: BasisSet

# --- Atomic-orbital evaluation (moved from Polynomials4ML.AtomicOrbitals,
# v0.6.1, and restructured to a generic Rnl·Ylm form). Built on the P4ML
# AbstractP4MLBasis interface; P4ML stays a dependency for the evaluation/AD/Lux
# machinery, SpheriCart supplies the angular Ylm. Evaluation is KernelAbstractions
# based on both CPU and GPU backends.
import Polynomials4ML: AbstractP4MLBasis,
                       _valtype, _gradtype,
                       _init_luxparams, _init_luxstate, pullback_ps,
                       _generate_input
# NB: `_static_params`/`_static_state` are intentionally NOT imported — this
# package owns them (see utils.jl), so we can give them `Any` fallbacks.
import ACEbase: evaluate, evaluate_ed, natural_indices
import ChainRulesCore: rrule, NoTangent, unthunk
using SpheriCart: SolidHarmonics
using LinearAlgebra: norm
using Random: AbstractRNG
# DecoratedParticles supplies the `PState` input type (position `x.𝐫` + species
# `x.S`) and its tangent type `VState`; a core dependency for now (an extension is
# a later option).
using DecoratedParticles: PState, VState

const ang2bohr = 1.8897261246257702

include("units.jl")
include("compiled_basis.jl")
include("adapt.jl")
include("kernels_2c.jl")
include("kernels_3c.jl")
include("reference/Reference.jl")
include("utils.jl")
include("orbitals/gtostoradials.jl")
include("orbitals/atomicorbitals.jl")
include("orbitals/utils.jl")

export compile_basis, adapt_basis,
       batch_overlap!, batch_overlap,
       batch_overlap_3c!, batch_overlap_3c

# atomic-orbital evaluation API (user-facing only)
export evaluate, evaluate_ed
export gaussian_orbitals, slater_orbitals

end # module AtomicOrbitalKernels
