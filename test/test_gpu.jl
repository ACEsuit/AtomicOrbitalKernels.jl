# Device tests for the 2-center / 3-center overlap kernels. The backend (`dev` /
# `dev_name`) comes from utils_gpu.jl; with no functional GPU, `dev = identity`
# and these run on the CPU backend (in Float32).

using AtomicOrbitalKernels
using AtomicOrbitalKernels: Reference
using Random
using Unitful
include(joinpath(@__DIR__, "utils_gpu.jl"))

@testset "2C overlap ($(dev_name), Float32)" begin
    rng = MersenneTwister(101)
    bc = compile_basis(BS_SI_DEFSVP)
    N = bc.nbf_total
    B = 64

    posA_raw, posA_u = random_positions(B; rng=rng)
    posB_raw, posB_u = random_positions(B; offset=(1.5, 0.0, 0.0), rng=rng)

    out_ref = zeros(Float64, N, N, B)
    Reference.batch_S_pair_ref!(out_ref, BS_SI_DEFSVP, posA_raw, posB_raw)

    bc_dev = adapt_basis(bc, dev, Float32)
    out_dev = dev(zeros(Float32, N, N, B))
    batch_overlap!(out_dev, bc_dev, posA_u, posB_u)
    out_h = Array(out_dev)
    @test maximum(abs, out_ref .- out_h) < 1f-3
end

@testset "3C overlap ($(dev_name), Float32)" begin
    rng = MersenneTwister(202)
    bc = compile_basis(BS_SI_DEFSVP)
    N = bc.nbf_total
    B = 4    # 3C tensor: N^3 * B Float32 ~= a few MB

    posA_raw, posA_u = random_positions(B; rng=rng)
    posB_raw, posB_u = random_positions(B; offset=(1.5, 0.0, 0.0), rng=rng)
    posC_raw, posC_u = random_positions(B; offset=(0.7, 1.2, 0.3), rng=rng)

    out_ref = zeros(Float64, N, N, N, B)
    Reference.batch_V_triple_ref!(out_ref, BS_SI_DEFSVP, posA_raw, posB_raw, posC_raw)

    bc_dev = adapt_basis(bc, dev, Float32)
    out_dev = dev(zeros(Float32, N, N, N, B))
    batch_overlap_3c!(out_dev, bc_dev, posA_u, posB_u, posC_u)
    out_h = Array(out_dev)
    @test maximum(abs, out_ref .- out_h) < 1f-3
end
