using AtomicOrbitalKernels
using AtomicOrbitalKernels: CompiledBasis

@testset "compile_basis: Si def2-SVP" begin
    bc = compile_basis(BS_SI_DEFSVP)
    @test bc isa CompiledBasis
    @test bc.nshells == length(BS_SI_DEFSVP.basis)
    @test bc.nbf_total == sum(((s.l + 1) * (s.l + 2) ÷ 2) for s in BS_SI_DEFSVP.basis)
    @test bc.Lmax == maximum(s.l for s in BS_SI_DEFSVP.basis)
    # prefix-sum invariants
    @test bc.basis_offset[1] == 0
    @test bc.basis_offset[end] == bc.nbf_total
    @test bc.prim_offset[1] == 0
    @test bc.prim_offset[end] == sum(length(s.coef) for s in BS_SI_DEFSVP.basis)
    @test length(bc.coef) == bc.prim_offset[end]
    @test length(bc.α)    == bc.prim_offset[end]

    # element types
    @test eltype(bc.coef) === Float64
    @test eltype(bc.α)    === Float64

    # T = Float32 produces a Float32 basis
    bc32 = compile_basis(BS_SI_DEFSVP, Float32)
    @test eltype(bc32.coef) === Float32
    @test eltype(bc32.α)    === Float32
end

@testset "compile_basis: H sto-3g" begin
    bc = compile_basis(BS_H_STO3G)
    @test bc.nshells == 1
    @test bc.Lmax    == 0
    @test bc.nbf_total == 1
end

@testset "adapt_basis: Array round-trip" begin
    bc = compile_basis(BS_SI_DEFSVP)
    bc2 = adapt_basis(bc, Array, Float64)
    @test bc2.nshells   == bc.nshells
    @test bc2.nbf_total == bc.nbf_total
    @test bc2.ls           == bc.ls
    @test bc2.nprim        == bc.nprim
    @test bc2.prim_offset  == bc.prim_offset
    @test bc2.coef         == bc.coef
    @test bc2.α            == bc.α
    @test bc2.basis_offset == bc.basis_offset

    # Cast to Float32
    bc32 = adapt_basis(bc, Array, Float32)
    @test eltype(bc32.coef) === Float32
    @test bc32.coef ≈ Float32.(bc.coef)
end
