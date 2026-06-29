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
# operator up to a basis-function permutation P (`S_A = P S_B Pᵀ`). We recover P
# by matching shells on (l, exponents) — within a shell the Cartesian component
# order is identical, since both paths run the same contraction kernel — and check
# the overlaps agree *elementwise* under P. That is strictly stronger than matching
# singular values, which only certify unitary equivalence (`S_A = U S_B V`), not a
# permutation, so a wrong coefficient or a mis-assigned shell could slip through.

using AtomicOrbitalKernels
import AtomicOrbitalKernels as AOK
using GaussianBasis: BasisSet
using AtomsBase: ChemicalSpecies
using DecoratedParticles: PState
using StaticArrays, Random, Unitful, Test

# Bohr positions → vector of PStates (orbital path); same points → Unitful Å
# matrix (GaussianBasis path).
_pstates(rawB, sp; T = Float64) =
        [ PState(𝐫 = SVector{3, T}(rawB[1, i], rawB[2, i], rawB[3, i]),
                 S = sp isa AbstractVector ? sp[i] : sp) for i in 1:size(rawB, 2) ]
_unitful(rawB) = (rawB ./ AOK.ang2bohr) .* u"angstrom"
_rawB(rng, B; offset = (0.0, 0.0, 0.0), scale = 0.9) =
        (randn(rng, 3, B) .* scale) .+ collect(offset)

# --- structural comparison: orbital path `cob::CartesianGTOBasis` (species 1)
# vs GaussianBasis path `bcc::CompiledBasis`. A cob shell's real primitives are
# the slots with nonzero coef (the rest are species padding). ---
_cob_real(cob, k)  = findall(!iszero, view(cob.coef, k, :, 1))
_cob_exps(cob, k)  = cob.ζ[k, _cob_real(cob, k), 1]
_cob_coef(cob, k)  = cob.coef[k, _cob_real(cob, k), 1]
_bcc_rng(bcc, j)   = bcc.prim_offset[j]+1 : bcc.prim_offset[j]+bcc.nprim[j]
_bcc_exps(bcc, j)  = bcc.α[_bcc_rng(bcc, j)]
_bcc_coef(bcc, j)  = bcc.coef[_bcc_rng(bcc, j)]

# Basis-function permutation P (`P[i]` = GB-path index of orbital-path bf `i`):
# match shells on (l, sorted exponents); within a shell the component order is
# identical (same contraction kernel). Returns (P, shell map σ). Errors if a shell
# can't be matched — i.e. the two bases are NOT permutation-equivalent.
function _bfn_perm(cob, bcc)
    @assert cob.nshells == bcc.nshells
    used = falses(bcc.nshells);  σ = zeros(Int, cob.nshells)
    for k in 1:cob.nshells
        ek = sort(_cob_exps(cob, k))
        j = findfirst(jj -> !used[jj] && bcc.ls[jj] == cob.ls[k] &&
                            length(_bcc_exps(bcc, jj)) == length(ek) &&
                            sort(_bcc_exps(bcc, jj)) ≈ ek, 1:bcc.nshells)
        j === nothing &&
            error("orbital shell $k (l=$(cob.ls[k])) has no GaussianBasis match")
        used[j] = true;  σ[k] = j
    end
    P = Int[]
    for k in 1:cob.nshells, c in 1:cob.nbf[k]
        push!(P, bcc.basis_offset[σ[k]] + c)
    end
    return P, σ
end

# matched shells must carry the same (exp, coef) primitives (sorted by exponent)
function _assert_shells_match(cob, bcc, σ)
    for k in 1:cob.nshells
        pc = sortperm(_cob_exps(cob, k));  pg = sortperm(_bcc_exps(bcc, σ[k]))
        @test _cob_exps(cob, k)[pc] ≈ _bcc_exps(bcc, σ[k])[pg]
        @test _cob_coef(cob, k)[pc] ≈ _bcc_coef(bcc, σ[k])[pg]
    end
end

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

            # ζ (exponents) stored verbatim; D is NOT stored — overlap gradients
            # flow back through the differentiable compile map, not a stored D
            @test Array(cob.ζ) == Array(orb.Rnl.ζ)

            # recover the permutation and confirm the two bases are the same
            # shells reordered (errors here if they are not permutation-equivalent)
            P, σ = _bfn_perm(cob, bcc)
            _assert_shells_match(cob, bcc, σ)

            rng = MersenneTwister(0xACE)
            rA = _rawB(rng, 4)
            rB = _rawB(rng, 4; offset = (3.0, 0.0, 0.0))
            SA = batch_overlap(cob, _pstates(rA, sp), _pstates(rB, sp))
            SB = batch_overlap(bcc, _unitful(rA), _unitful(rB))
            @test all(isfinite, SA)
            # off-diagonal (A≠B): overlaps agree elementwise under P (S_A = P S_B Pᵀ)
            for b in 1:size(SA, 3)
                @test SA[:, :, b] ≈ SB[P, P, b] atol = 1e-10
            end

            o = zeros(3, 1)                  # self-overlap (A=B)
            selc = batch_overlap(cob, _pstates(o, sp), _pstates(o, sp))[:, :, 1]
            selg = batch_overlap(bcc, _unitful(o), _unitful(o))[:, :, 1]
            @test selc ≈ selg[P, P] atol = 1e-10
        end
    end
end

@testset "Float32 compile + overlap" begin
    sp  = ChemicalSpecies(8)
    # element type is inferred from (ζ,D); build a Float32 orbital basis
    orb32 = gaussian_orbitals(BasisSet("cc-pvdz", "O 0.0 0.0 0.0"); T = Float32)
    cob32 = compile_basis(orb32)
    @test eltype(cob32.coef) == Float32
    rng = MersenneTwister(1)
    rA = _rawB(rng, 3);  rB = _rawB(rng, 3; offset = (3.0, 0.0, 0.0))
    # output precision is taken from the input positions
    S32 = batch_overlap(cob32, _pstates(rA, sp; T = Float32), _pstates(rB, sp; T = Float32))
    cob64 = compile_basis(gaussian_orbitals(BasisSet("cc-pvdz", "O 0.0 0.0 0.0")))
    S64 = batch_overlap(cob64, _pstates(rA, sp), _pstates(rB, sp))
    @test eltype(S32) == Float32
    # same orbital spec ordering in both → compare elementwise
    @test maximum(abs, S32 - S64) < 1e-4
end

@testset "batch_overlap from orbital basis (compile-on-call)" begin
    sp  = ChemicalSpecies(8)
    orb = gaussian_orbitals(BasisSet("cc-pvdz", "O 0.0 0.0 0.0"))
    cob = compile_basis(orb)
    rng = MersenneTwister(3)
    rA = _rawB(rng, 4);  rB = _rawB(rng, 4; offset = (3.0, 0.0, 0.0))
    XA = _pstates(rA, sp);  XB = _pstates(rB, sp)
    ref = batch_overlap(cob, XA, XB)
    @test batch_overlap(orb, XA, XB) ≈ ref atol = 1e-12
    out = zeros(cob.nbf_total, cob.nbf_total, length(XA))
    batch_overlap!(out, orb, XA, XB)
    @test out ≈ ref atol = 1e-12
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
    # exactly (same l-ascending spec → same ordering), so compare elementwise
    for (sp, el) in zip(sps, ("C", "N", "O"))
        single = compile_basis(gaussian_orbitals(BasisSet("cc-pvdz", "$el 0.0 0.0 0.0")))
        Sm = batch_overlap(cob,    _pstates(rA, sp), _pstates(rB, sp))
        Ss = batch_overlap(single, _pstates(rA, sp), _pstates(rB, sp))
        @test Sm ≈ Ss atol = 1e-10
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
