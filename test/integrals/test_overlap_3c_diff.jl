# Differentiable 3-center overlap. The Lux-style
# `batch_overlap(orb, XA, XB, XC, ps, st)` recomputes `coef` from the radial
# params `ps.Rnl = (ζ, D)`, so the overlap is differentiable w.r.t. `(ζ, D)`. The
# analytic parameter pullback (`pullback_ps` / `rrule`) is checked against
# ForwardDiff directional derivatives, mirroring test_overlap_2c_diff.jl. (The
# normalization pullback `_compile_coef_pb` is shared with 2C and tested there.)

using AtomicOrbitalKernels
import AtomicOrbitalKernels as AOK
using GaussianBasis: BasisSet
using AtomsBase: ChemicalSpecies
using DecoratedParticles: PState
import LuxCore
import ChainRulesCore
using ForwardDiff: Dual, partials
import ForwardDiff
using StaticArrays, Random, Test

_dps3(raw, sp; T = Float64) =
        [ PState(𝐫 = SVector{3, T}(raw[1, i], raw[2, i], raw[3, i]),
                 S = sp isa AbstractVector ? sp[i] : sp) for i in 1:size(raw, 2) ]

@testset "single-species (6-31g / C)" begin
    rng = MersenneTwister(7)
    orb = gaussian_orbitals(BasisSet("6-31g", "C 0.0 0.0 0.0");
                            length_unit = :bohr)
    sp  = ChemicalSpecies(6)
    B = 2
    rA = randn(rng, 3, B) .* 0.5
    rB = randn(rng, 3, B) .* 0.5;  rB[1, :] .+= 2.0
    rC = randn(rng, 3, B) .* 0.5;  rC[2, :] .+= 1.0
    XA = _dps3(rA, sp);  XB = _dps3(rB, sp);  XC = _dps3(rC, sp)
    ps = LuxCore.initialparameters(rng, orb)
    st = LuxCore.initialstates(rng, orb)

    V = batch_overlap(orb, XA, XB, XC, ps, st)
    # the Lux path equals the static compile-on-call and precompiled paths
    @test V ≈ batch_overlap(orb, XA, XB, XC)
    @test V ≈ batch_overlap(compile_basis(orb), XA, XB, XC)

    N = size(V, 1)
    ∂V = randn(rng, N, N, N, B)
    ∂ps = AOK.pullback_ps(∂V, orb, XA, XB, XC, ps, st)
    @test size(∂ps.Rnl.ζ) == size(ps.Rnl.ζ)
    @test size(∂ps.Rnl.D) == size(ps.Rnl.D)

    # parameter pullback vs ForwardDiff directional derivatives
    ζ = ps.Rnl.ζ;  D = ps.Rnl.D
    W = randn(rng, size(D));  Vd = randn(rng, size(ζ))
    lossD(Di) = sum(∂V .* batch_overlap(orb, XA, XB, XC,
                            (Rnl = (ζ = ζ, D = Di), Ylm = ps.Ylm), st))
    lossζ(ζi) = sum(∂V .* batch_overlap(orb, XA, XB, XC,
                            (Rnl = (ζ = ζi, D = D), Ylm = ps.Ylm), st))
    @test sum(∂ps.Rnl.D .* W) ≈ partials(lossD(D .+ Dual(0.0, 1.0) .* W), 1)
    @test sum(∂ps.Rnl.ζ .* Vd) ≈ partials(lossζ(ζ .+ Dual(0.0, 1.0) .* Vd), 1)

    # rrule round-trip: value matches, params match the pullback, positions are
    # NoTangent (parameters-only differentiation)
    y, pb = AOK.rrule(batch_overlap, orb, XA, XB, XC, ps, st)
    @test y ≈ V
    ∂f, ∂orb, ∂XA, ∂XB, ∂XC, ∂ps_r, ∂st = pb(∂V)
    @test ∂XA isa ChainRulesCore.NoTangent
    @test ∂XB isa ChainRulesCore.NoTangent
    @test ∂XC isa ChainRulesCore.NoTangent
    @test ∂ps_r.Rnl.D ≈ ∂ps.Rnl.D
    @test ∂ps_r.Rnl.ζ ≈ ∂ps.Rnl.ζ
end

@testset "multi-species + absent species" begin
    rng = MersenneTwister(4321)
    zlist = (6, 1, 8)
    # K=3: a real contraction, so `coef` (and hence the overlap) genuinely
    # depends on `D` — for K=1 `coef = D/√normsq` is scale-invariant and ∂D ≡ 0.
    orb = gaussian_orbitals(3, 2; length_unit = :bohr, nspecies = 3,
                            zlist = zlist, K = 3)
    ps = LuxCore.initialparameters(rng, orb)
    st = LuxCore.initialstates(rng, orb)

    B = 3
    spA = rand(rng, zlist, B);  spB = rand(rng, zlist, B)
    spC = rand(rng, zlist, B)
    XA = _dps3(randn(rng, 3, B) .* 0.5, spA)
    XB = _dps3(randn(rng, 3, B) .* 0.5 .+ [1.5, 0.0, 0.0], spB)
    XC = _dps3(randn(rng, 3, B) .* 0.5 .+ [0.5, 1.0, 0.0], spC)

    V = batch_overlap(orb, XA, XB, XC, ps, st)
    @test V ≈ batch_overlap(compile_basis(orb), XA, XB, XC)
    N = size(V, 1)
    ∂V = randn(rng, N, N, N, B)
    ∂ps = AOK.pullback_ps(∂V, orb, XA, XB, XC, ps, st)
    ζ = ps.Rnl.ζ;  D = ps.Rnl.D
    Vd = randn(rng, size(ζ));  W = randn(rng, size(D))
    lossD(Di) = sum(∂V .* batch_overlap(orb, XA, XB, XC,
                            (Rnl = (ζ = ζ, D = Di), Ylm = ps.Ylm), st))
    lossζ(ζi) = sum(∂V .* batch_overlap(orb, XA, XB, XC,
                            (Rnl = (ζ = ζi, D = D), Ylm = ps.Ylm), st))
    @test sum(∂ps.Rnl.D .* W) ≈ partials(lossD(D .+ Dual(0.0, 1.0) .* W), 1)
    @test sum(∂ps.Rnl.ζ .* Vd) ≈ partials(lossζ(ζ .+ Dual(0.0, 1.0) .* Vd), 1)

    # a species absent from the batch gets a zero parameter gradient
    XAs = _dps3(randn(rng, 3, B) .* 0.5, rand(rng, (6, 1), B))
    XBs = _dps3(randn(rng, 3, B) .* 0.5 .+ [1.5, 0.0, 0.0], rand(rng, (6, 1), B))
    XCs = _dps3(randn(rng, 3, B) .* 0.5 .+ [0.5, 1.0, 0.0], rand(rng, (6, 1), B))
    ∂ps_s = AOK.pullback_ps(randn(rng, N, N, N, B), orb, XAs, XBs, XCs, ps, st)
    σ8 = AOK._z2i(orb.Rnl, 8)
    @test all(iszero, ∂ps_s.Rnl.ζ[:, :, σ8])
    @test all(iszero, ∂ps_s.Rnl.D[:, :, σ8])
end

@testset "Float32" begin
    rng = MersenneTwister(9)
    orb = gaussian_orbitals(3, 2; length_unit = :bohr, T = Float32)
    ps = LuxCore.initialparameters(rng, orb)
    st = LuxCore.initialstates(rng, orb)
    B = 2
    XA = _dps3(randn(rng, 3, B) .* 0.5f0, 1; T = Float32)
    XB = _dps3(randn(rng, 3, B) .* 0.5f0 .+ [1.5f0, 0, 0], 1; T = Float32)
    XC = _dps3(randn(rng, 3, B) .* 0.5f0 .+ [0, 1.0f0, 0], 1; T = Float32)
    V = batch_overlap(orb, XA, XB, XC, ps, st)
    @test eltype(V) == Float32
    ∂ps = AOK.pullback_ps(Float32.(randn(rng, size(V))), orb, XA, XB, XC, ps, st)
    @test eltype(∂ps.Rnl.ζ) == Float32
    @test all(isfinite, ∂ps.Rnl.ζ) && all(isfinite, ∂ps.Rnl.D)
end
