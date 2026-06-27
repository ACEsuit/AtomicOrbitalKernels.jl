# Validate `compile_basis(::AtomicOrbitals)` (species-aware Cartesian) against the
# GaussianBasis Cartesian path: `BasisSet(name, el)` (spherical) → `gaussian_orbitals`
# → `compile_basis` should reproduce `BasisSet(name, el; spherical=false)` →
# `compile_basis`, both run through the overlap kernel.
#
# The orbital overlap takes `PState` inputs (position `x.𝐫` in atomic units / Bohr
# + species `x.S`), mirroring `evaluate(::AtomicOrbitals, X)`; the GaussianBasis
# path takes `Unitful` positions. To feed both the SAME physical points, a Bohr
# position matrix `rawB` is passed to the orbital path verbatim and as
# `rawB/ang2bohr · u"angstrom"` to the GaussianBasis path (so its `to_bohr` returns
# `rawB`).
#
# The orbital radial spec is l-ascending while GaussianBasis shell order can
# interleave l (e.g. 6-31g = s,s,p,s,p), so the two paths give the same overlap
# operator up to a basis-function permutation P (`S_A = P S_B Pᵀ`). Singular
# values are invariant under that (and equal the eigenvalues for the symmetric,
# PSD self-overlap), so we compare sorted `svdvals` — order-independent.

using AtomicOrbitalKernels
import AtomicOrbitalKernels as AOK
using GaussianBasis: BasisSet
using AtomsBase: ChemicalSpecies
using DecoratedParticles: PState
using StaticArrays, LinearAlgebra, Random, Unitful, Test

# Bohr positions → vector of PStates (orbital path); same points → Unitful Å
# matrix (GaussianBasis path).
_pstates(rawB, sp) =
        [ PState(𝐫 = SVector{3, Float64}(rawB[1, i], rawB[2, i], rawB[3, i]),
                 S = sp isa AbstractVector ? sp[i] : sp) for i in 1:size(rawB, 2) ]
_unitful(rawB) = (rawB ./ AOK.ang2bohr) .* u"angstrom"
_rawB(rng, B; offset = (0.0, 0.0, 0.0), scale = 0.9) =
        (randn(rng, 3, B) .* scale) .+ collect(offset)

@testset "AtomicOrbitals → Cartesian vs GaussianBasis Cartesian" begin
    for (name, el, Z) in (("sto-3g", "H", 1), ("6-31g", "C", 6), ("cc-pvdz", "O", 8),
                          ("def2-svp", "O", 8), ("cc-pvtz", "N", 7))
        @testset "$name / $el" begin
            sp  = ChemicalSpecies(Z)
            orb = gaussian_orbitals(BasisSet(name, "$el 0.0 0.0 0.0"))
            cob = compile_basis(orb)
            bcc = compile_basis(BasisSet(name, "$el 0.0 0.0 0.0"; spherical = false))

            @test cob.zlist == (sp,)
            @test AOK.nspecies(cob) == 1
            @test cob.nbf_total == bcc.nbf_total
            @test sort(cob.ls) == sort(bcc.ls)           # same shells, maybe reordered

            # (ζ,D) stored verbatim → parameters transfer back to the orbital basis
            @test Array(cob.ζ) == Array(orb.Rnl.ζ)
            @test Array(cob.D) == Array(orb.Rnl.D)

            rng = MersenneTwister(0xACE)
            rA = _rawB(rng, 4)
            rB = _rawB(rng, 4; offset = (3.0, 0.0, 0.0))
            SA = batch_overlap(cob, _pstates(rA, sp), _pstates(rB, sp))
            SB = batch_overlap(bcc, _unitful(rA), _unitful(rB))
            @test all(isfinite, SA)
            for b in 1:size(SA, 3)           # off-diagonal (A≠B): singular values
                @test sort(svdvals(SA[:, :, b])) ≈ sort(svdvals(SB[:, :, b])) atol = 1e-10
            end

            o = zeros(3, 1)                  # self-overlap (A=B): PSD ⇒ svd = eig
            selc = batch_overlap(cob, _pstates(o, sp), _pstates(o, sp))[:, :, 1]
            selg = batch_overlap(bcc, _unitful(o), _unitful(o))[:, :, 1]
            @test sort(svdvals(selc)) ≈ sort(svdvals(selg)) atol = 1e-10
        end
    end
end

@testset "Float32 compile + overlap" begin
    sp  = ChemicalSpecies(8)
    orb = gaussian_orbitals(BasisSet("cc-pvdz", "O 0.0 0.0 0.0"))
    cob32 = compile_basis(orb, Float32)
    @test eltype(cob32.coef) == Float32
    rng = MersenneTwister(1)
    rA = _rawB(rng, 3);  rB = _rawB(rng, 3; offset = (3.0, 0.0, 0.0))
    XA = _pstates(rA, sp);  XB = _pstates(rB, sp)
    S32 = batch_overlap(cob32, XA, XB; FT = Float32)
    S64 = batch_overlap(compile_basis(orb), XA, XB)
    @test eltype(S32) == Float32
    for b in 1:size(S32, 3)
        @test sort(svdvals(S32[:, :, b])) ≈ sort(svdvals(S64[:, :, b])) atol = 1e-4
    end
end

@testset "multi-species slices" begin
    # cc-pvdz C, N, O share the same 3s2p1d structure → one shared spec, no padding
    orb = gaussian_orbitals("cc-pvdz", [:C, :N, :O])
    cob = compile_basis(orb)
    sps = (ChemicalSpecies(6), ChemicalSpecies(7), ChemicalSpecies(8))
    @test AOK.nspecies(cob) == 3
    @test cob.zlist == sps

    rng = MersenneTwister(7)
    rA = _rawB(rng, 4);  rB = _rawB(rng, 4; offset = (3.0, 0.0, 0.0))

    # each species slice reproduces that element's single-element converter
    for (sp, el) in zip(sps, ("C", "N", "O"))
        single = compile_basis(gaussian_orbitals(BasisSet("cc-pvdz", "$el 0.0 0.0 0.0")))
        Sm = batch_overlap(cob,    _pstates(rA, sp), _pstates(rB, sp))
        Ss = batch_overlap(single, _pstates(rA, sp), _pstates(rB, sp))
        for b in 1:size(Sm, 3)
            @test sort(svdvals(Sm[:, :, b])) ≈ sort(svdvals(Ss[:, :, b])) atol = 1e-10
        end
    end

    # mixed species pair (C bra, O ket) runs, finite, right shape
    Smix = batch_overlap(cob, _pstates(rA, sps[1]), _pstates(rB, sps[3]))
    @test size(Smix) == (cob.nbf_total, cob.nbf_total, size(rA, 2))
    @test all(isfinite, Smix)
end

@testset "errors" begin
    # Slater radials carry a radial power the overlap kernels can't integrate
    @test_throws ErrorException compile_basis(slater_orbitals(3, 2))

    cob = compile_basis(gaussian_orbitals(BasisSet("sto-3g", "H 0.0 0.0 0.0")))
    # a species not in the basis errors
    bad = [ PState(𝐫 = SVector(0.0, 0.0, 0.0), S = ChemicalSpecies(2)) ]   # He ∉ zlist
    @test_throws ErrorException batch_overlap(cob, bad, bad)
    # mismatched batch lengths error
    g = PState(𝐫 = SVector(0.0, 0.0, 0.0), S = ChemicalSpecies(1))
    @test_throws ErrorException batch_overlap(cob, [g], [g, g])
end
