# Independent Gauss–Hermite quadrature oracle for the Cartesian overlap kernels
# (Stage 5). Unlike the `Reference` oracle — which shares the kernels' block
# layout — this integrates the Cartesian AO products directly and writes blocks
# with the correct `nbf` stride, so it is a true ground truth independent of the
# McMurchie–Davidson recursion and the block indexing (GH is exact for these
# polynomial×Gaussian integrands). It was introduced to catch the l ≥ 2 write-
# stride aliasing; with the stride fixed to `nbf`, kernel and oracle now agree on
# the full matrix, including the d/f blocks.

using AtomicOrbitalKernels
using AtomicOrbitalKernels: Reference
using Random
using Unitful
using Test

include("quad_oracle.jl")

# basis-function indices belonging to l ≤ 1 shells (no aliasing there)
function _low_l_idx(BS)
    idx = Int[]
    off = 0
    for s in BS.basis
        nbf = (s.l + 1) * (s.l + 2) ÷ 2
        s.l ≤ 1 && append!(idx, off+1:off+nbf)
        off += nbf
    end
    return idx
end

@testset "oracle self-checks" begin
    # s/s closed form: two normalized-free s-Gaussians overlap to (π/(α+β))^1.5
    t, w = _gh_nodes(0)
    a, b = 1.3, 0.7
    Sx = _quad_S1d(0, 0, a, b, 0.0, 0.0, t, w)
    @test Sx^3 ≈ (π / (a + b))^1.5

    # on an Lmax ≤ 1 basis the oracle must agree with Reference (both correct)
    rng = MersenneTwister(11)
    N = compile_basis(BS_H_STO3G).nbf_total
    B = 3
    posA_raw, _ = random_positions(B; scale=0.3, rng=rng)
    posB_raw, _ = random_positions(B; offset=(0.74, 0.0, 0.0), scale=0.3, rng=rng)
    out_ref = zeros(Float64, N, N, B)
    Reference.batch_S_pair_ref!(out_ref, BS_H_STO3G, posA_raw, posB_raw)
    out_q = zeros(Float64, N, N, B)
    quad_batch_S_pair!(out_q, BS_H_STO3G, posA_raw, posB_raw)
    @test maximum(abs, out_q .- out_ref) < 1e-10
end

@testset "2-center: kernel vs quadrature oracle" begin
    rng = MersenneTwister(0x5715)

    # Lmax = 0 (H sto-3g): kernel and oracle agree everywhere
    bcH = compile_basis(BS_H_STO3G)
    NH = bcH.nbf_total
    B = 3
    pA_raw, pA_u = random_positions(B; scale=0.3, rng=rng)
    pB_raw, pB_u = random_positions(B; offset=(0.74, 0.0, 0.0), scale=0.3, rng=rng)
    ka = zeros(Float64, NH, NH, B); batch_overlap!(ka, bcH, pA_u, pB_u)
    qo = zeros(Float64, NH, NH, B); quad_batch_S_pair!(qo, BS_H_STO3G, pA_raw, pB_raw)
    @test maximum(abs, ka .- qo) < 1e-10

    # Lmax = 2 (Si def2-SVP, has a d shell): exercises the d block whose `2l+1`
    # write stride used to scramble it. The l ≤ 1 sub-block is checked separately
    # as a more localized regression signal.
    bc = compile_basis(BS_SI_DEFSVP)
    N = bc.nbf_total
    qA_raw, qA_u = random_positions(B; offset=(0.0, 0.0, 0.0), scale=0.5, rng=rng)
    qB_raw, qB_u = random_positions(B; offset=(1.5, 0.0, 0.0), scale=0.5, rng=rng)
    ka2 = zeros(Float64, N, N, B); batch_overlap!(ka2, bc, qA_u, qB_u)
    qo2 = zeros(Float64, N, N, B); quad_batch_S_pair!(qo2, BS_SI_DEFSVP, qA_raw, qB_raw)

    lo = _low_l_idx(BS_SI_DEFSVP)
    @test maximum(abs, ka2[lo, lo, :] .- qo2[lo, lo, :]) < 1e-9

    d_full = maximum(abs, ka2 .- qo2)
    @info "2C Si def2-SVP: max |kernel − oracle| over the full matrix" d_full
    # full matrix (incl. the d block) now agrees — the stride fix removed the
    # l ≥ 2 aliasing
    @test d_full < 1e-9
end

@testset "3-center: kernel vs quadrature oracle" begin
    rng = MersenneTwister(0x3C3C)
    bc = compile_basis(BS_SI_DEFSVP)
    N = bc.nbf_total
    B = 2
    pA_raw, pA_u = random_positions(B; offset=(0.0, 0.0, 0.0), scale=0.5, rng=rng)
    pB_raw, pB_u = random_positions(B; offset=(1.5, 0.0, 0.0), scale=0.5, rng=rng)
    pC_raw, pC_u = random_positions(B; offset=(0.7, 1.2, 0.3), scale=0.5, rng=rng)

    ka = zeros(Float64, N, N, N, B)
    batch_overlap_3c!(ka, bc, pA_u, pB_u, pC_u)
    qo = zeros(Float64, N, N, N, B)
    quad_batch_V_triple!(qo, BS_SI_DEFSVP, pA_raw, pB_raw, pC_raw)

    lo = _low_l_idx(BS_SI_DEFSVP)
    @test maximum(abs, ka[lo, lo, lo, :] .- qo[lo, lo, lo, :]) < 1e-8

    d_full = maximum(abs, ka .- qo)
    @info "3C Si def2-SVP: max |kernel − oracle| over the full tensor" d_full
    # full tensor (incl. the d block) now agrees — the stride fix removed the
    # l ≥ 2 aliasing
    @test d_full < 1e-8
end
