using GaussianBasisKernels
using GaussianBasisKernels: Reference
using Random
using Unitful

@testset "batch_overlap! vs Reference (Si def2-SVP, Lmax=2)" begin
    rng = MersenneTwister(0xC0DE)
    bc = compile_basis(BS_SI_DEFSVP)
    N = bc.nbf_total

    for B in (1, 4)
        posA_raw, posA_u = random_positions(B; offset=(0.0, 0.0, 0.0), scale=0.5, rng=rng)
        posB_raw, posB_u = random_positions(B; offset=(1.5, 0.0, 0.0), scale=0.5, rng=rng)

        out_ref = zeros(Float64, N, N, B)
        Reference.batch_S_pair_ref!(out_ref, BS_SI_DEFSVP, posA_raw, posB_raw)

        out_ka = zeros(Float64, N, N, B)
        batch_overlap!(out_ka, bc, posA_u, posB_u)

        @test maximum(abs, out_ka .- out_ref) < 1e-10
    end
end

@testset "batch_overlap! vs Reference (H sto-3g, Lmax=0)" begin
    rng = MersenneTwister(0xBEEF)
    bc = compile_basis(BS_H_STO3G)
    N = bc.nbf_total

    B = 3
    posA_raw, posA_u = random_positions(B; scale=0.3, rng=rng)
    posB_raw, posB_u = random_positions(B; offset=(0.74, 0.0, 0.0), scale=0.3, rng=rng)

    out_ref = zeros(Float64, N, N, B)
    Reference.batch_S_pair_ref!(out_ref, BS_H_STO3G, posA_raw, posB_raw)

    out_ka = zeros(Float64, N, N, B)
    batch_overlap!(out_ka, bc, posA_u, posB_u)

    @test maximum(abs, out_ka .- out_ref) < 1e-12

    # Self-overlap (A == B) is positive on the diagonal
    out_self = zeros(Float64, N, N, B)
    batch_overlap!(out_self, bc, posA_u, posA_u)
    @test all(out_self[i, i, 1] > 0 for i in 1:N)
end

@testset "batch_overlap allocating wrapper" begin
    bc = compile_basis(BS_H_STO3G)
    B = 2
    _, posA_u = random_positions(B; rng=MersenneTwister(1))
    _, posB_u = random_positions(B; offset=(1.0, 0.0, 0.0), rng=MersenneTwister(2))

    out = batch_overlap(bc, posA_u, posB_u)
    @test size(out) == (bc.nbf_total, bc.nbf_total, B)
    @test eltype(out) === Float64
end

@testset "batch_overlap! rejects plain-Real positions" begin
    bc = compile_basis(BS_H_STO3G)
    B = 2
    out = zeros(Float64, bc.nbf_total, bc.nbf_total, B)
    plain = randn(3, B)
    unit  = randn(3, B) .* u"angstrom"
    @test_throws ArgumentError batch_overlap!(out, bc, plain, unit)
    @test_throws ArgumentError batch_overlap!(out, bc, unit, plain)
    @test_throws ArgumentError batch_overlap!(out, bc, plain, plain)
end

@testset "batch_overlap! accepts u\"nm\" (unit conversion)" begin
    rng = MersenneTwister(42)
    bc = compile_basis(BS_H_STO3G)
    B = 2

    _, posA_ang = random_positions(B; rng=rng)
    _, posB_ang = random_positions(B; offset=(1.0, 0.0, 0.0), rng=rng)
    # Same physical positions expressed in nm (1 nm = 10 Å). Must give the
    # identical result through the unit-conversion path.
    posA_nm = (ustrip.(posA_ang) ./ 10) .* u"nm"
    posB_nm = (ustrip.(posB_ang) ./ 10) .* u"nm"

    out_ang = batch_overlap(bc, posA_ang, posB_ang)
    out_nm  = batch_overlap(bc, posA_nm,  posB_nm)
    @test out_ang ≈ out_nm  atol=1e-12
end
