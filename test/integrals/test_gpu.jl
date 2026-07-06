# Device tests for the 2-center / 3-center overlap kernels. The backend (`dev` /
# `gpu_backend` / `gpu_supports_f64`) comes from ACETestUtils; with no functional
# GPU, `dev = identity` and these run on the CPU backend.
#
# Every testset runs in Float32 and, when the backend supports it (not Metal),
# again in Float64 against the same CPU Float64 reference at a tight tolerance.

using AtomicOrbitalKernels
using AtomicOrbitalKernels: Reference
using GaussianBasis: BasisSet
using AtomsBase: ChemicalSpecies
using DecoratedParticles: PState
using StaticArrays
using Random
using Unitful
using ACETestUtils: gpu_test_backend
(; dev, gpu_backend, gpu_supports_f64) = gpu_test_backend()

# `dev` (e.g. `CUDA.cu`) downcasts Float64 → Float32, so precision-controlled
# device arrays are built through the device array *type* instead.
_DevArr = dev === identity ? Array : typeof(dev(zeros(Float32, 1))).name.wrapper
FTs = gpu_supports_f64 ? (Float32, Float64) : (Float32,)
_gputol(FT) = FT === Float32 ? 1f-3 : 1e-10

@testset "2C overlap ($(gpu_backend), $FT)" for FT in FTs
    rng = MersenneTwister(101)
    bc = compile_basis(BS_SI_DEFSVP)
    N = bc.nbf_total
    B = 64

    posA_raw, posA_u = random_positions(B; rng=rng)
    posB_raw, posB_u = random_positions(B; offset=(1.5, 0.0, 0.0), rng=rng)

    out_ref = zeros(Float64, N, N, B)
    Reference.batch_S_pair_ref!(out_ref, BS_SI_DEFSVP, posA_raw, posB_raw)

    bc_dev = adapt_basis(bc, _DevArr, FT)
    out_dev = _DevArr(zeros(FT, N, N, B))
    batch_overlap!(out_dev, bc_dev, posA_u, posB_u)
    out_h = Array(out_dev)
    @test maximum(abs, out_ref .- out_h) < _gputol(FT)
end

@testset "3C overlap ($(gpu_backend), $FT)" for FT in FTs
    rng = MersenneTwister(202)
    bc = compile_basis(BS_SI_DEFSVP)
    N = bc.nbf_total
    B = 4    # 3C tensor: N^3 * B ~= a few MB

    posA_raw, posA_u = random_positions(B; rng=rng)
    posB_raw, posB_u = random_positions(B; offset=(1.5, 0.0, 0.0), rng=rng)
    posC_raw, posC_u = random_positions(B; offset=(0.7, 1.2, 0.3), rng=rng)

    out_ref = zeros(Float64, N, N, N, B)
    Reference.batch_V_triple_ref!(out_ref, BS_SI_DEFSVP, posA_raw, posB_raw, posC_raw)

    bc_dev = adapt_basis(bc, _DevArr, FT)
    out_dev = _DevArr(zeros(FT, N, N, N, B))
    batch_overlap_3c!(out_dev, bc_dev, posA_u, posB_u, posC_u)
    out_h = Array(out_dev)
    @test maximum(abs, out_ref .- out_h) < _gputol(FT)
end

# --- orbital path (species-aware CartesianGTOBasis, PState inputs); the
# device result is checked against the same kernel run on the CPU in Float64 ---

_ps_dev(raw, sp) =
    [ PState(𝐫 = SVector{3,Float64}(raw[1, i], raw[2, i], raw[3, i]), S = sp)
      for i in 1:size(raw, 2) ]

@testset "2C orbital overlap ($(gpu_backend), $FT)" for FT in FTs
    rng = MersenneTwister(303)
    orb = gaussian_orbitals(BasisSet("def2-svp", "Si 0.0 0.0 0.0");
                            length_unit = :bohr)
    cob = compile_basis(orb)
    sp  = ChemicalSpecies(14)
    N   = cob.nbf_total
    B   = 64
    rA = randn(rng, 3, B) .* 0.5
    rB = randn(rng, 3, B) .* 0.5;  rB[1, :] .+= 1.5
    XA = _ps_dev(rA, sp);  XB = _ps_dev(rB, sp)
    ref = batch_overlap(cob, XA, XB)              # CPU, Float64

    cob_dev = adapt_basis(cob, _DevArr, FT)
    out_dev = _DevArr(zeros(FT, N, N, B))
    batch_overlap!(out_dev, cob_dev, XA, XB)
    @test maximum(abs, ref .- Array(out_dev)) < _gputol(FT)
end

@testset "3C orbital overlap ($(gpu_backend), $FT)" for FT in FTs
    rng = MersenneTwister(404)
    orb = gaussian_orbitals(BasisSet("def2-svp", "Si 0.0 0.0 0.0");
                            length_unit = :bohr)
    cob = compile_basis(orb)
    sp  = ChemicalSpecies(14)
    N   = cob.nbf_total
    B   = 4    # 3C tensor: N^3 * B ~= a few MB
    rA = randn(rng, 3, B) .* 0.5
    rB = randn(rng, 3, B) .* 0.5;  rB[1, :] .+= 1.5
    rC = randn(rng, 3, B) .* 0.5;  rC[2, :] .+= 1.0
    XA = _ps_dev(rA, sp);  XB = _ps_dev(rB, sp);  XC = _ps_dev(rC, sp)
    ref = batch_overlap(cob, XA, XB, XC)          # CPU, Float64

    cob_dev = adapt_basis(cob, _DevArr, FT)
    out_dev = _DevArr(zeros(FT, N, N, N, B))
    batch_overlap!(out_dev, cob_dev, XA, XB, XC)
    @test maximum(abs, ref .- Array(out_dev)) < _gputol(FT)
end