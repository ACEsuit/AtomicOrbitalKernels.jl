# Opt-in GPU tests. Skipped silently if no CUDA or Metal backend is functional.

using AtomicOrbitalKernels
using AtomicOrbitalKernels: Reference
import AtomicOrbitalKernels as AOK
using Pkg
using Random
using Unitful
using StaticArrays
using LinearAlgebra

const _DEPS = keys(Pkg.project().dependencies)

_gpu = nothing
_gpu_name = ""

if "CUDA" in _DEPS
    try
        @eval Main using CUDA
        if Main.CUDA.functional()
            global _gpu = Main.cu
            global _gpu_name = "CUDA"
        end
    catch e
        @info "CUDA load failed: $(sprint(showerror, e))"
    end
end

if _gpu === nothing && "Metal" in _DEPS
    try
        @eval Main using Metal
        if Main.Metal.functional()
            global _gpu = Main.mtl
            global _gpu_name = "Metal"
        end
    catch e
        @info "Metal load failed: $(sprint(showerror, e))"
    end
end

if _gpu === nothing
    @info "No GPU backend available — skipping GPU tests."
else
    @testset "GPU 2C ($(_gpu_name), Float32)" begin
        rng = MersenneTwister(101)
        bc = compile_basis(BS_SI_DEFSVP)
        N = bc.nbf_total
        B = 64

        posA_raw, posA_u = random_positions(B; rng=rng)
        posB_raw, posB_u = random_positions(B; offset=(1.5, 0.0, 0.0), rng=rng)

        out_ref = zeros(Float64, N, N, B)
        Reference.batch_S_pair_ref!(out_ref, BS_SI_DEFSVP, posA_raw, posB_raw)

        bc_gpu = adapt_basis(bc, _gpu, Float32)
        out_gpu = _gpu(zeros(Float32, N, N, B))
        batch_overlap!(out_gpu, bc_gpu, posA_u, posB_u)
        out_h = Array(out_gpu)
        @test maximum(abs, out_ref .- out_h) < 1f-3
    end

    @testset "GPU 3C ($(_gpu_name), Float32)" begin
        rng = MersenneTwister(202)
        bc = compile_basis(BS_SI_DEFSVP)
        N = bc.nbf_total
        B = 4    # 3C tensor: N^3 * B Float32 ~= a few MB

        posA_raw, posA_u = random_positions(B; rng=rng)
        posB_raw, posB_u = random_positions(B; offset=(1.5, 0.0, 0.0), rng=rng)
        posC_raw, posC_u = random_positions(B; offset=(0.7, 1.2, 0.3), rng=rng)

        out_ref = zeros(Float64, N, N, N, B)
        Reference.batch_V_triple_ref!(out_ref, BS_SI_DEFSVP, posA_raw, posB_raw, posC_raw)

        bc_gpu = adapt_basis(bc, _gpu, Float32)
        out_gpu = _gpu(zeros(Float32, N, N, N, B))
        batch_overlap_3c!(out_gpu, bc_gpu, posA_u, posB_u, posC_u)
        out_h = Array(out_gpu)
        @test maximum(abs, out_ref .- out_h) < 1f-3
    end

    @testset "GPU AtomicOrbitals eval ($(_gpu_name), Float32)" begin
        basis = gaussian_orbitals()
        st = (Rnl = NamedTuple(), Ylm = (Flm = basis.Ylm.Flm,))

        Xh = [@SVector randn(3) for _ = 1:64]
        Pc = evaluate_ref(basis, Xh)                  # CPU Float64 reference (forward)
        _, dPc = evaluate_ed(basis, Xh)               # CPU KA gradient (Float64)

        Xg = _gpu([SVector{3, Float32}(x) for x in Xh])
        psg = (Rnl = (ζ = _gpu(Float32.(basis.Rnl.ζ)), D = _gpu(Float32.(basis.Rnl.D))),
               Ylm = NamedTuple())

        Pg = evaluate(basis, Xg, psg, st)
        @test !(Pg isa Array)                        # ran on the device
        @test norm(Array(Pg) .- Float32.(Pc)) / norm(Pc) < 1f-3

        _, dPg = evaluate_ed(basis, Xg, psg, st)
        dPh = Array(dPg)
        num = maximum(norm(dPh[i] - SVector{3, Float32}(dPc[i])) for i in eachindex(dPh))
        den = maximum(norm(SVector{3, Float32}(v)) for v in dPc)
        @test num / den < 1f-2
    end
end
