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
# v0.6.1). Concrete Rnl·Ylm orbital types built on the P4ML AbstractP4MLBasis
# interface; P4ML stays a dependency for radial building blocks (MonoBasis) and
# the evaluation/AD/Lux machinery. SpheriCart supplies the angular Ylm.
import Polynomials4ML: AbstractP4MLBasis, MonoBasis, BATCH,
                       _valtype, _evaluate!, _static_params,
                       _init_luxparams, _init_luxstate, pullback_ps,
                       _generate_input
import ACEbase: evaluate, evaluate_ed, natural_indices
using SpheriCart: SolidHarmonics, ComplexSolidHarmonics, ComplexSphericalHarmonics
using LinearAlgebra: norm
using Random: AbstractRNG

const ang2bohr = 1.8897261246257702

include("units.jl")
include("compiled_basis.jl")
include("adapt.jl")
include("kernels_2c.jl")
include("kernels_3c.jl")
include("reference/Reference.jl")
include("orbitals/atomicorbitals.jl")

export compile_basis, adapt_basis,
       batch_overlap!, batch_overlap,
       batch_overlap_3c!, batch_overlap_3c

# atomic-orbital evaluation API
export AtomicOrbitals, RadialDecay, GaussianDecay, SlaterDecay
export evaluate, evaluate_ed

end # module AtomicOrbitalKernels
