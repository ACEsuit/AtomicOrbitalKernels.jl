# Validate the species-aware 3-center overlap (`batch_overlap(cob, XA, XB, XC)`)
# against the Gauss–Hermite quadrature oracle, driven directly on the
# `CartesianGTOBasis`: `_quad_triple_block!` is duck-typed over shells with
# fields `(l, coef, exp)`, so we can feed it the compiled per-species primitives
# without going through a GaussianBasis `BasisSet`. That avoids the shell
# permutation between the l-ascending orbital spec and GaussianBasis shell
# order, and — unlike the GB path — covers mixed-species batches.
#
# A GB-path cross-check is still included for sto-3g C, whose shell order
# (s, s, p) happens to be l-ascending, so the two paths agree elementwise
# with no permutation (asserted via `ls` before comparing).

using AtomicOrbitalKernels
import AtomicOrbitalKernels as AOK
using GaussianBasis: BasisSet
using AtomsBase: ChemicalSpecies
using DecoratedParticles: PState
using StaticArrays, Random, Unitful, Test

include("quad_oracle.jl")   # _gh_nodes, _quad_triple_block!

_ps3(raw, sp) =
    [ PState(𝐫 = SVector{3,Float64}(raw[1, i], raw[2, i], raw[3, i]),
             S = sp isa AbstractVector ? sp[i] : sp) for i in 1:size(raw, 2) ]
_raw3(rng, B; offset = (0.0, 0.0, 0.0)) =
    (randn(rng, 3, B) .* 0.9) .+ collect(offset)

# shell `k` of species-slice `σ` as a duck-typed (l, coef, exp) oracle shell;
# the real primitives are the slots with nonzero coef (the rest are padding)
function _cob_shell(cob, k, σ)
    idx = findall(!iszero, view(cob.coef, k, :, σ))
    return (l = cob.ls[k], coef = cob.coef[k, idx, σ], exp = cob.ζ[k, idx, σ])
end

# GH-oracle ground truth for the orbital-path 3C overlap: same PState inputs
# and per-element species resolution as `batch_overlap`, positions scaled by
# `lengthscale` into Bohr, blocks written as 3D views (correct nbf strides).
function quad_V_orbital(cob, XA, XB, XC)
    N = cob.nbf_total
    B = length(XA)
    out = zeros(N, N, N, B)
    t, w = _gh_nodes(cob.Lmax)
    off = cob.basis_offset
    s = cob.lengthscale
    for b in 1:B
        σa = findfirst(==(XA[b].S), cob.zlist)
        σb = findfirst(==(XB[b].S), cob.zlist)
        σc = findfirst(==(XC[b].S), cob.zlist)
        A = SVector{3,Float64}(XA[b].𝐫) .* s
        Bv = SVector{3,Float64}(XB[b].𝐫) .* s
        C = SVector{3,Float64}(XC[b].𝐫) .* s
        for s_a in 1:cob.nshells, s_b in 1:cob.nshells, s_c in 1:cob.nshells
            block = view(out, off[s_a]+1:off[s_a]+cob.nbf[s_a],
                              off[s_b]+1:off[s_b]+cob.nbf[s_b],
                              off[s_c]+1:off[s_c]+cob.nbf[s_c], b)
            _quad_triple_block!(block, _cob_shell(cob, s_a, σa),
                                _cob_shell(cob, s_b, σb),
                                _cob_shell(cob, s_c, σc), A, Bv, C, t, w)
        end
    end
    return out
end

@testset "orbital 3C vs GH oracle (single species)" begin
    # 6-31g C: mixed primitive counts (6,3,3,1,1 → K=6) exercises the
    # zero-coef padding skip; def2-svp O has a d shell (l=2, strided writes)
    for (name, el, Z) in (("6-31g", "C", 6), ("def2-svp", "O", 8))
        @testset "$name / $el" begin
            sp  = ChemicalSpecies(Z)
            orb = gaussian_orbitals(BasisSet(name, "$el 0.0 0.0 0.0");
                                    length_unit = :bohr)
            cob = compile_basis(orb)
            rng = MersenneTwister(0x3C)
            XA = _ps3(_raw3(rng, 2), sp)
            XB = _ps3(_raw3(rng, 2; offset = (2.0, 0.0, 0.0)), sp)
            XC = _ps3(_raw3(rng, 2; offset = (0.7, 1.2, 0.3)), sp)
            V  = batch_overlap(cob, XA, XB, XC)
            Vq = quad_V_orbital(cob, XA, XB, XC)
            @test all(isfinite, V)
            @test V ≈ Vq atol = 1e-10
        end
    end
end

@testset "orbital 3C vs GaussianBasis path (sto-3g C)" begin
    orb = gaussian_orbitals(BasisSet("sto-3g", "C 0.0 0.0 0.0");
                            length_unit = :bohr)
    cob = compile_basis(orb)
    bcc = compile_basis(BasisSet("sto-3g", "C 0.0 0.0 0.0"; spherical = false))
    # sto-3g C shell order (s, s, p) is already l-ascending → no permutation
    @test cob.ls == bcc.ls
    rng = MersenneTwister(0xC3)
    rA = _raw3(rng, 3)
    rB = _raw3(rng, 3; offset = (2.0, 0.0, 0.0))
    rC = _raw3(rng, 3; offset = (0.7, 1.2, 0.3))
    sp = ChemicalSpecies(6)
    V_orb = batch_overlap(cob, _ps3(rA, sp), _ps3(rB, sp), _ps3(rC, sp))
    _uf(r) = (r ./ AOK.ang2bohr) .* u"angstrom"
    V_gb  = batch_overlap_3c(bcc, _uf(rA), _uf(rB), _uf(rC))
    @test V_orb ≈ V_gb atol = 1e-10
end

@testset "mixed species (cc-pvdz C,N,O)" begin
    orb = gaussian_orbitals("cc-pvdz", [:C, :N, :O]; length_unit = :bohr)
    cob = compile_basis(orb)
    sps = (ChemicalSpecies(6), ChemicalSpecies(7), ChemicalSpecies(8))
    rng = MersenneTwister(77)
    rA = _raw3(rng, 2)
    rB = _raw3(rng, 2; offset = (2.0, 0.0, 0.0))
    rC = _raw3(rng, 2; offset = (0.7, 1.2, 0.3))

    # uniform species triple (C, N, O) and species varying WITHIN the batch
    XA = _ps3(rA, [sps[1], sps[3]])
    XB = _ps3(rB, [sps[2], sps[1]])
    XC = _ps3(rC, [sps[3], sps[2]])
    V  = batch_overlap(cob, XA, XB, XC)
    Vq = quad_V_orbital(cob, XA, XB, XC)
    @test size(V) == (cob.nbf_total, cob.nbf_total, cob.nbf_total, 2)
    @test V ≈ Vq atol = 1e-10

    # each species slice reproduces that element's single-element compile
    for (sp, el) in zip(sps, ("C", "N", "O"))
        single = compile_basis(gaussian_orbitals(
                BasisSet("cc-pvdz", "$el 0.0 0.0 0.0"); length_unit = :bohr))
        Vm = batch_overlap(cob,    _ps3(rA, sp), _ps3(rB, sp), _ps3(rC, sp))
        Vs = batch_overlap(single, _ps3(rA, sp), _ps3(rB, sp), _ps3(rC, sp))
        @test Vm ≈ Vs atol = 1e-10
    end
end

@testset "compile-on-call, in-place, errors" begin
    sp  = ChemicalSpecies(6)
    orb = gaussian_orbitals(BasisSet("6-31g", "C 0.0 0.0 0.0");
                            length_unit = :bohr)
    cob = compile_basis(orb)
    rng = MersenneTwister(5)
    XA = _ps3(_raw3(rng, 2), sp)
    XB = _ps3(_raw3(rng, 2; offset = (2.0, 0.0, 0.0)), sp)
    XC = _ps3(_raw3(rng, 2; offset = (0.7, 1.2, 0.3)), sp)
    ref = batch_overlap(cob, XA, XB, XC)
    @test batch_overlap(orb, XA, XB, XC) ≈ ref atol = 1e-12
    N = cob.nbf_total
    out = zeros(N, N, N, 2)
    batch_overlap!(out, orb, XA, XB, XC)
    @test out ≈ ref atol = 1e-12

    # mismatched batch lengths error
    @test_throws ErrorException batch_overlap(cob, XA, XB, XC[1:1])
    # a species not in the basis errors
    bad = _ps3(_raw3(rng, 2), ChemicalSpecies(2))    # He ∉ zlist
    @test_throws ErrorException batch_overlap(cob, XA, XB, bad)
end
