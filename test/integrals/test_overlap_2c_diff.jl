# Differentiable 2-center overlap (Stage 4). The Lux-style
# `batch_overlap(orb, XA, XB, ps, st)` recomputes `coef` from the radial params
# `ps.Rnl = (ζ, D)`, so the overlap is differentiable w.r.t. `(ζ, D)`. The analytic
# parameter pullback (`pullback_ps` / `rrule`) is checked against ForwardDiff
# directional derivatives, exactly as the orbital `evaluate` tests are.

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

# Bohr positions (a 3×B matrix) → vector of PStates; `sp` is one species or a
# per-point species vector.
_dpstates(raw, sp; T = Float64) =
        [ PState(𝐫 = SVector{3, T}(raw[1, i], raw[2, i], raw[3, i]),
                 S = sp isa AbstractVector ? sp[i] : sp) for i in 1:size(raw, 2) ]

@testset "single-species (6-31g / C)" begin
    rng = MersenneTwister(7)
    orb = gaussian_orbitals(BasisSet("6-31g", "C 0.0 0.0 0.0"); length_unit = :bohr)
    sp  = ChemicalSpecies(6)
    B = 4
    rA = randn(rng, 3, B) .* 0.5
    rB = randn(rng, 3, B) .* 0.5;  rB[1, :] .+= 2.0
    XA = _dpstates(rA, sp);  XB = _dpstates(rB, sp)
    ps = LuxCore.initialparameters(rng, orb)
    st = LuxCore.initialstates(rng, orb)

    S = batch_overlap(orb, XA, XB, ps, st)
    # the Lux path equals the static compile-on-call and the precompiled paths
    @test S ≈ batch_overlap(orb, XA, XB)
    @test S ≈ batch_overlap(compile_basis(orb), XA, XB)

    N = size(S, 1)
    ∂S = randn(rng, N, N, B)
    ∂ps = AOK.pullback_ps(∂S, orb, XA, XB, ps, st)
    @test size(∂ps.Rnl.ζ) == size(ps.Rnl.ζ)
    @test size(∂ps.Rnl.D) == size(ps.Rnl.D)

    # parameter pullback vs ForwardDiff directional derivatives (D enters only via
    # the normalization, ζ via both the kernel exponent and the normalization)
    ζ = ps.Rnl.ζ;  D = ps.Rnl.D
    V = randn(rng, size(ζ));  W = randn(rng, size(D))
    lossD(Di) = sum(∂S .* batch_overlap(orb, XA, XB,
                            (Rnl = (ζ = ζ, D = Di), Ylm = ps.Ylm), st))
    lossζ(ζi) = sum(∂S .* batch_overlap(orb, XA, XB,
                            (Rnl = (ζ = ζi, D = D), Ylm = ps.Ylm), st))
    @test sum(∂ps.Rnl.D .* W) ≈ partials(lossD(D .+ Dual(0.0, 1.0) .* W), 1)
    @test sum(∂ps.Rnl.ζ .* V) ≈ partials(lossζ(ζ .+ Dual(0.0, 1.0) .* V), 1)

    # rrule round-trip: value matches, params match the pullback, positions are
    # NoTangent (parameters-only differentiation)
    y, pb = AOK.rrule(batch_overlap, orb, XA, XB, ps, st)
    @test y ≈ S
    ∂f, ∂orb, ∂XA, ∂XB, ∂ps_r, ∂st = pb(∂S)
    @test ∂XA isa ChainRulesCore.NoTangent
    @test ∂XB isa ChainRulesCore.NoTangent
    @test ∂ps_r.Rnl.D ≈ ∂ps.Rnl.D
    @test ∂ps_r.Rnl.ζ ≈ ∂ps.Rnl.ζ
end

@testset "normalization pullback (cc-pvdz / O)" begin
    rng = MersenneTwister(2)
    orb = gaussian_orbitals(BasisSet("cc-pvdz", "O 0.0 0.0 0.0"); length_unit = :bohr)
    ps  = AOK._static_params(orb)
    ζ = ps.Rnl.ζ;  D = ps.Rnl.D
    ls  = AOK._compile_struct(orb).ls
    coef0 = AOK._compile_coef(ζ, D, ls)
    ∂coef = randn(rng, size(coef0))
    ∂D, ∂ζ = AOK._compile_coef_pb(∂coef, ζ, D, ls)

    W = randn(rng, size(D));  V = randn(rng, size(ζ))
    fD(t) = sum(∂coef .* AOK._compile_coef(ζ, D .+ t .* W, ls))
    fζ(t) = sum(∂coef .* AOK._compile_coef(ζ .+ t .* V, D, ls))
    @test sum(∂D .* W) ≈ ForwardDiff.derivative(fD, 0.0)
    @test sum(∂ζ .* V) ≈ ForwardDiff.derivative(fζ, 0.0)
end

@testset "multi-species + absent species" begin
    rng = MersenneTwister(4321)
    zlist = (6, 1, 8)
    # K=3: a real contraction, so `coef` (and hence the overlap) genuinely depends
    # on `D` — for K=1 `coef = D/√normsq` is scale-invariant and ∂/∂D ≡ 0.
    orb = gaussian_orbitals(4, 3; length_unit = :bohr, nspecies = 3,
                            zlist = zlist, K = 3)
    ps = LuxCore.initialparameters(rng, orb)
    st = LuxCore.initialstates(rng, orb)
    @test ps.Rnl.ζ isa Array && size(ps.Rnl.ζ, 3) == 3

    B = 12
    spA = rand(rng, zlist, B);  spB = rand(rng, zlist, B)
    XA = _dpstates(randn(rng, 3, B) .* 0.5, spA)
    XB = _dpstates(randn(rng, 3, B) .* 0.5 .+ [1.5, 0.0, 0.0], spB)

    S = batch_overlap(orb, XA, XB, ps, st)
    @test S ≈ batch_overlap(compile_basis(orb), XA, XB)
    N = size(S, 1)
    ∂S = randn(rng, N, N, B)
    ∂ps = AOK.pullback_ps(∂S, orb, XA, XB, ps, st)
    ζ = ps.Rnl.ζ;  D = ps.Rnl.D
    V = randn(rng, size(ζ));  W = randn(rng, size(D))
    lossD(Di) = sum(∂S .* batch_overlap(orb, XA, XB,
                            (Rnl = (ζ = ζ, D = Di), Ylm = ps.Ylm), st))
    lossζ(ζi) = sum(∂S .* batch_overlap(orb, XA, XB,
                            (Rnl = (ζ = ζi, D = D), Ylm = ps.Ylm), st))
    @test sum(∂ps.Rnl.D .* W) ≈ partials(lossD(D .+ Dual(0.0, 1.0) .* W), 1)
    @test sum(∂ps.Rnl.ζ .* V) ≈ partials(lossζ(ζ .+ Dual(0.0, 1.0) .* V), 1)

    # a species absent from the batch gets a zero parameter gradient
    XAs = _dpstates(randn(rng, 3, B) .* 0.5, rand(rng, (6, 1), B))
    XBs = _dpstates(randn(rng, 3, B) .* 0.5 .+ [1.5, 0.0, 0.0], rand(rng, (6, 1), B))
    ∂ps_s = AOK.pullback_ps(randn(rng, N, N, B), orb, XAs, XBs, ps, st)
    σ8 = AOK._z2i(orb.Rnl, 8)
    @test all(iszero, ∂ps_s.Rnl.ζ[:, :, σ8])
    @test all(iszero, ∂ps_s.Rnl.D[:, :, σ8])
end

@testset "Float32" begin
    rng = MersenneTwister(9)
    orb = gaussian_orbitals(3, 2; length_unit = :bohr, T = Float32)
    ps = LuxCore.initialparameters(rng, orb)
    st = LuxCore.initialstates(rng, orb)
    B = 5
    XA = _dpstates(randn(rng, 3, B) .* 0.5f0, 1; T = Float32)
    XB = _dpstates(randn(rng, 3, B) .* 0.5f0 .+ [1.5f0, 0, 0], 1; T = Float32)
    S = batch_overlap(orb, XA, XB, ps, st)
    @test eltype(S) == Float32
    ∂ps = AOK.pullback_ps(Float32.(randn(rng, size(S))), orb, XA, XB, ps, st)
    @test eltype(∂ps.Rnl.ζ) == Float32
    @test all(isfinite, ∂ps.Rnl.ζ) && all(isfinite, ∂ps.Rnl.D)
end
