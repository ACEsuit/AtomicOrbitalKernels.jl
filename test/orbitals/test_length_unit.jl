# Length-unit plumbing. `length_unit` is required at construction. A basis built
# with a non-Bohr unit evaluates / overlaps at the same physical points as the
# `:bohr` basis fed positions scaled to Bohr — the parameters stay native (Bohr),
# so the two agree exactly. The spatial gradient picks up the unit factor (chain
# rule), cross-checked against ForwardDiff.

using AtomicOrbitalKernels
import AtomicOrbitalKernels as AOK
using AtomicOrbitalKernels: evaluate, evaluate_ed, evaluate_ref
using GaussianBasis: BasisSet
using AtomsBase: ChemicalSpecies
using DecoratedParticles: PState
using ForwardDiff: Dual, partials
using StaticArrays, LinearAlgebra, Random, Unitful, Test

const a2b = AOK.ang2bohr

@testset "length_unit is required (no default)" begin
    @test_throws UndefKeywordError gaussian_orbitals(BasisSet("sto-3g", "H 0 0 0"))
    @test_throws UndefKeywordError gaussian_orbitals(4, 3)
    @test_throws UndefKeywordError gaussian_orbitals("sto-3g", [:H])
    @test_throws UndefKeywordError slater_orbitals(3, 2)
end

@testset "lengthscale factors" begin
    f(u) = gaussian_orbitals(BasisSet("sto-3g", "H 0 0 0"); length_unit = u).lengthscale
    @test f(:bohr) == 1.0
    @test f(:angstrom) ≈ a2b
    @test f(:Å) ≈ a2b
    @test f(u"angstrom") ≈ a2b
    @test f(u"nm") ≈ 10 * a2b
    @test_throws ErrorException f(:furlong)
end

@testset "evaluate / gradient / overlap unit-equivalence" begin
    rng = MersenneTwister(11)
    bs   = BasisSet("cc-pvdz", "O 0 0 0")
    orbB = gaussian_orbitals(bs; length_unit = :bohr)
    orbA = gaussian_orbitals(bs; length_unit = :angstrom)
    X  = [ @SVector randn(3) for _ = 1:10 ]      # interpreted as Å by orbA
    Xb = [ a2b .* x for x in X ]                 # same physical points, in Bohr

    # values agree exactly (parameters are native Bohr in both)
    @test evaluate(orbA, X) ≈ evaluate(orbB, Xb) atol = 1e-12

    # spatial gradient: ∂/∂(Å input) = a2b · ∂/∂(Bohr input)
    _, dA = evaluate_ed(orbA, X)
    _, dB = evaluate_ed(orbB, Xb)
    @test maximum(maximum(abs, dA[i] .- a2b .* dB[i]) for i in eachindex(dA)) < 1e-12

    # gradient factor cross-checked against ForwardDiff on evaluate_ref(orbA, ·)
    Nb = length(orbA)
    U  = [ @SVector randn(3) for _ in eachindex(X) ]
    Xd = [ X[i] + Dual(0.0, 1.0) * U[i] for i in eachindex(X) ]
    Yd = evaluate_ref(orbA, Xd)
    dY = [ partials(Yd[i, n], 1) for i in eachindex(X), n = 1:Nb ]
    dU = [ dot(dA[i, n], U[i])   for i in eachindex(X), n = 1:Nb ]
    @test dY ≈ dU

    # the overlap inherits the unit via compile_basis
    cobA = compile_basis(orbA);  cobB = compile_basis(orbB)
    @test cobA.lengthscale == orbA.lengthscale
    sp  = ChemicalSpecies(8)
    XA  = [ PState(𝐫 = x, S = sp) for x in X ]
    XBp = [ PState(𝐫 = (a2b .* x), S = sp) for x in X ]
    @test batch_overlap(cobA, XA, XA) ≈ batch_overlap(cobB, XBp, XBp) atol = 1e-12
end
