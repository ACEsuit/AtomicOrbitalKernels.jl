using AtomicOrbitalKernels
using AtomicOrbitalKernels: Reference
using Random
using Unitful

@testset "batch_overlap_3c! vs Reference (Si def2-SVP, Lmax=2)" begin
    rng = MersenneTwister(0xD00D)
    bc = compile_basis(BS_SI_DEFSVP)
    N = bc.nbf_total

    B = 2   # 3C is O(N^3 * B), keep B small
    posA_raw, posA_u = random_positions(B; offset=(0.0, 0.0, 0.0), scale=0.5, rng=rng)
    posB_raw, posB_u = random_positions(B; offset=(1.5, 0.0, 0.0), scale=0.5, rng=rng)
    posC_raw, posC_u = random_positions(B; offset=(0.7, 1.2, 0.3), scale=0.5, rng=rng)

    out_ref = zeros(Float64, N, N, N, B)
    Reference.batch_V_triple_ref!(out_ref, BS_SI_DEFSVP, posA_raw, posB_raw, posC_raw)

    out_ka = zeros(Float64, N, N, N, B)
    batch_overlap_3c!(out_ka, bc, posA_u, posB_u, posC_u)

    @test maximum(abs, out_ka .- out_ref) < 1e-10
end

@testset "batch_overlap_3c! vs Reference (H sto-3g)" begin
    rng = MersenneTwister(0xFACE)
    bc = compile_basis(BS_H_STO3G)
    N = bc.nbf_total
    B = 3

    posA_raw, posA_u = random_positions(B; scale=0.3, rng=rng)
    posB_raw, posB_u = random_positions(B; offset=(0.7, 0.0, 0.0), scale=0.3, rng=rng)
    posC_raw, posC_u = random_positions(B; offset=(0.0, 0.6, 0.0), scale=0.3, rng=rng)

    out_ref = zeros(Float64, N, N, N, B)
    Reference.batch_V_triple_ref!(out_ref, BS_H_STO3G, posA_raw, posB_raw, posC_raw)

    out_ka = zeros(Float64, N, N, N, B)
    batch_overlap_3c!(out_ka, bc, posA_u, posB_u, posC_u)

    @test maximum(abs, out_ka .- out_ref) < 1e-12
end

@testset "batch_overlap_3c allocating wrapper + nm unit input" begin
    rng = MersenneTwister(7)
    bc = compile_basis(BS_H_STO3G)
    B = 2
    _, posA_u = random_positions(B; rng=rng)
    _, posB_u = random_positions(B; offset=(0.7, 0.0, 0.0), rng=rng)
    _, posC_u = random_positions(B; offset=(0.0, 0.6, 0.0), rng=rng)

    out_ang = batch_overlap_3c(bc, posA_u, posB_u, posC_u)

    # Same positions in nm (1 nm == 10 Å)
    posA_nm = (ustrip.(posA_u) ./ 10) .* u"nm"
    posB_nm = (ustrip.(posB_u) ./ 10) .* u"nm"
    posC_nm = (ustrip.(posC_u) ./ 10) .* u"nm"
    out_nm = batch_overlap_3c(bc, posA_nm, posB_nm, posC_nm)

    @test size(out_ang) == (bc.nbf_total, bc.nbf_total, bc.nbf_total, B)
    @test out_ang ≈ out_nm  atol=1e-12
end

@testset "batch_overlap_3c! rejects plain-Real positions" begin
    bc = compile_basis(BS_H_STO3G)
    B = 2
    out = zeros(Float64, bc.nbf_total, bc.nbf_total, bc.nbf_total, B)
    plain = randn(3, B)
    unit  = randn(3, B) .* u"angstrom"
    @test_throws ArgumentError batch_overlap_3c!(out, bc, plain, unit,  unit)
    @test_throws ArgumentError batch_overlap_3c!(out, bc, unit,  plain, unit)
    @test_throws ArgumentError batch_overlap_3c!(out, bc, unit,  unit,  plain)
end
