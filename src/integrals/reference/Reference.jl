"""
    AtomicOrbitalKernels.Reference

Pedagogical scalar implementation of the Cartesian-Gaussian 2- and 3-center
overlap integrals used as the correctness oracle for the batched
KernelAbstractions kernels. Mirrors the prototype's `prototype/micro/gaussints.jl`.

Not exported. Access via `AtomicOrbitalKernels.Reference.<name>`. Used by the
package test suite and available to downstream code that wants to sanity-check
the kernel output.
"""
module Reference

using StaticArrays
using GaussianBasis: CartesianShell
import Molecules

const ang2bohr = 1.8897261246257702

include("ecoeffs.jl")
include("shell_pair.jl")
include("shell_triple.jl")

end # module Reference
