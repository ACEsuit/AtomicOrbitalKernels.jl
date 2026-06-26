# Validate `compile_basis(::AtomicOrbitals)` (species-aware Cartesian) against the
# GaussianBasis Cartesian path: `BasisSet(name, el)` (spherical) → `gaussian_orbitals`
# → `compile_basis` should reproduce `BasisSet(name, el; spherical=false)` →
# `compile_basis`, both run through the same overlap kernel.
#
# The orbital radial spec is l-ascending while GaussianBasis shell order can
# interleave l (e.g. 6-31g lists s,s,p,s,p), so the two paths produce the same
# overlap operator up to a basis-function permutation P: `S_A = P S_B Pᵀ`.
# Singular values are invariant under that (and equal the eigenvalues for the
# symmetric, PSD self-overlap), so we compare sorted `svdvals` — order-independent.

using AtomicOrbitalKernels
import AtomicOrbitalKernels as AOK
using GaussianBasis: BasisSet
using AtomsBase: ChemicalSpecies
using StaticArrays, LinearAlgebra, Random, Unitful, Test

_uang(M) = M .* u"angstrom"
_randpos(rng, B; offset = (0.0, 0.0, 0.0), scale = 0.5) =
        _uang((randn(rng, 3, B) .* scale) .+ collect(offset))

@testset "AtomicOrbitals → Cartesian vs GaussianBasis Cartesian" begin
    for (name, el) in (("sto-3g", "H"), ("6-31g", "C"), ("cc-pvdz", "O"),
                       ("def2-svp", "O"), ("cc-pvtz", "N"))
        @testset "$name / $el" begin
            orb = gaussian_orbitals(BasisSet(name, "$el 0.0 0.0 0.0"))
            cob = compile_basis(orb)
            bcc = compile_basis(BasisSet(name, "$el 0.0 0.0 0.0"; spherical = false))

            @test AOK.nspecies(cob) == 1
            @test cob.nbf_total == bcc.nbf_total
            @test sort(cob.ls) == sort(bcc.ls)           # same shells, maybe reordered

            # (ζ,D) stored verbatim → parameters transfer back to the orbital basis
            @test Array(cob.ζ) == Array(orb.Rnl.ζ)
            @test Array(cob.D) == Array(orb.Rnl.D)

            rng = MersenneTwister(0xACE)
            posA = _randpos(rng, 4)
            posB = _randpos(rng, 4; offset = (1.6, 0.0, 0.0))
            SA = batch_overlap(cob, posA, posB)
            SB = batch_overlap(bcc, posA, posB)
            @test all(isfinite, SA)
            for b in 1:size(SA, 3)            # off-diagonal (A≠B): singular values
                @test sort(svdvals(SA[:, :, b])) ≈ sort(svdvals(SB[:, :, b])) atol = 1e-10
            end

            o = _uang(zeros(3, 1))           # self-overlap (A=B): PSD ⇒ svd = eig
            selc = batch_overlap(cob, o, o)[:, :, 1]
            selg = batch_overlap(bcc, o, o)[:, :, 1]
            @test sort(svdvals(selc)) ≈ sort(svdvals(selg)) atol = 1e-10
        end
    end
end

@testset "Float32 compile + overlap" begin
    orb = gaussian_orbitals(BasisSet("cc-pvdz", "O 0.0 0.0 0.0"))
    cob32 = compile_basis(orb, Float32)
    @test eltype(cob32.coef) == Float32
    rng = MersenneTwister(1)
    posA = _randpos(rng, 3);  posB = _randpos(rng, 3; offset = (1.5, 0.0, 0.0))
    S32 = batch_overlap(cob32, posA, posB; FT = Float32)
    S64 = batch_overlap(compile_basis(orb), posA, posB)
    @test eltype(S32) == Float32
    for b in 1:size(S32, 3)
        @test sort(svdvals(S32[:, :, b])) ≈ sort(svdvals(S64[:, :, b])) atol = 1e-4
    end
end

@testset "multi-species slices" begin
    # cc-pvdz C, N, O share the same 3s2p1d structure → one shared spec, no padding
    orb = gaussian_orbitals("cc-pvdz", [:C, :N, :O])
    cob = compile_basis(orb)
    @test AOK.nspecies(cob) == 3
    @test cob.zlist ==
          (ChemicalSpecies(6), ChemicalSpecies(7), ChemicalSpecies(8))

    rng = MersenneTwister(7)
    posA = _randpos(rng, 4);  posB = _randpos(rng, 4; offset = (1.5, 0.0, 0.0))
    B = size(posA, 2)

    # each species slice reproduces that element's single-element converter
    for (σ, el) in ((1, "C"), (2, "N"), (3, "O"))
        single = compile_basis(gaussian_orbitals(BasisSet("cc-pvdz", "$el 0.0 0.0 0.0")))
        Sm = batch_overlap(cob, posA, posB; sidxA = fill(σ, B), sidxB = fill(σ, B))
        Ss = batch_overlap(single, posA, posB)
        for b in 1:B
            @test sort(svdvals(Sm[:, :, b])) ≈ sort(svdvals(Ss[:, :, b])) atol = 1e-10
        end
    end

    # mixed species pair (C bra, O ket) runs, finite, right shape
    Smix = batch_overlap(cob, posA, posB; sidxA = fill(1, B), sidxB = fill(3, B))
    @test size(Smix) == (cob.nbf_total, cob.nbf_total, B)
    @test all(isfinite, Smix)
end

@testset "errors" begin
    # Slater radials carry a radial power the overlap kernels can't integrate
    @test_throws ErrorException compile_basis(slater_orbitals(3, 2))
    # positions must be Unitful
    cob = compile_basis(gaussian_orbitals(BasisSet("sto-3g", "H 0.0 0.0 0.0")))
    out = zeros(cob.nbf_total, cob.nbf_total, 2)
    @test_throws ArgumentError batch_overlap!(out, cob, randn(3, 2), randn(3, 2))
end
