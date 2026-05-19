using GaussianBasisKernels: Reference
using GaussianBasis
using Molecules
using StaticArrays

# Closed-form check: overlap of two un-normalised, same-center Gaussians,
# φ(r) = exp(-α r²), is (π/(α+β))^{3/2}. For two single-primitive s-shells
# with coefficient 1 and exponents α, β, `Reference.generate_S_pair!` should
# return exactly that.
@testset "Reference: s/s closed-form overlap" begin
    α = 1.3
    β = 0.7
    atom = Molecules.Atom(1, 1.008, SA[0.0, 0.0, 0.0])
    sa = GaussianBasis.CartesianShell(0, [1.0], [α], atom)
    sb = GaussianBasis.CartesianShell(0, [1.0], [β], atom)
    out = zeros(Float64, 1)
    E = Reference.alloc_E(sa, sb)
    Reference.generate_S_pair!(out, E, sa, sb)
    @test out[1] ≈ (π / (α + β))^1.5  atol = 1e-12
end

# Symmetry: S[i, j] ≈ S[j, i] (within numerical noise) for any non-trivial
# shell pair. Build a p-shell with two primitives, pair with itself displaced.
@testset "Reference: shell-pair symmetry" begin
    atom_A = Molecules.Atom(6, 12.0, SA[0.0, 0.0, 0.0])
    atom_B = Molecules.Atom(6, 12.0, SA[0.4, 0.7, -0.2])
    p_A = GaussianBasis.CartesianShell(1, [0.7, 0.3], [1.2, 0.4], atom_A)
    p_B = GaussianBasis.CartesianShell(1, [0.7, 0.3], [1.2, 0.4], atom_B)

    out_AB = zeros(Float64, 9)
    E_AB = Reference.alloc_E(p_A, p_B)
    Reference.generate_S_pair!(out_AB, E_AB, p_A, p_B)

    out_BA = zeros(Float64, 9)
    E_BA = Reference.alloc_E(p_B, p_A)
    Reference.generate_S_pair!(out_BA, E_BA, p_B, p_A)

    SAB = reshape(out_AB, 3, 3)
    SBA = reshape(out_BA, 3, 3)
    @test SAB ≈ SBA'  atol = 1e-12
end
