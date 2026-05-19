# Ported from prototype/micro/batch_bench.jl. Validates the package's
# `batch_overlap!` / `batch_overlap_3c!` against the bundled scalar reference
# and prints timings for the CPU and (if available) GPU backends.
#
# Run from the repository root:
#   julia --project=bench bench/batch_bench.jl
#
# To enable a GPU backend, add CUDA or Metal to bench/Project.toml first.

using GaussianBasisKernels
using GaussianBasisKernels: Reference
using GaussianBasis
using Molecules
using StaticArrays
using BenchmarkTools
using KernelAbstractions
using Random
using Unitful

## --- basis set --------------------------------------------------------------

const SI = Molecules.Atom(14, 28.0855, SA[0.0, 0.0, 0.0])
BS = BasisSet("def2-SVP", [SI]; spherical=false, lib=:acsint)
bc = compile_basis(BS)
N = bc.nbf_total
@info "Basis: $(bc.nshells) shells, $N basis fns (cartesian), Lmax=$(bc.Lmax)"

## --- 2-center batch ---------------------------------------------------------

Random.seed!(0xC0DE)
B = 1024
posA_raw = randn(3, B) .* 0.5
posB_raw = randn(3, B) .* 0.5 .+ SA[1.5, 0.0, 0.0]
posA = posA_raw .* u"angstrom"
posB = posB_raw .* u"angstrom"

out_ref = zeros(Float64, N, N, B)
Reference.batch_S_pair_ref!(out_ref, BS, posA_raw, posB_raw)

out_cpu = zeros(Float64, N, N, B)
batch_overlap!(out_cpu, bc, posA, posB)
max_err = maximum(abs, out_ref .- out_cpu)
@info "2C max |Δ| (CPU vs Reference) = $max_err"
@assert max_err < 1e-10

print("2C naive ref  (B=$B): "); @btime Reference.batch_S_pair_ref!($out_ref, $BS, $posA_raw, $posB_raw)
print("2C KA  / CPU  (B=$B): "); @btime batch_overlap!($out_cpu, $bc, $posA, $posB)

## --- GPU 2C -----------------------------------------------------------------

using Pkg
const _DEPS = keys(Pkg.project().dependencies)

gpu = nothing
gpu_name = ""

if "CUDA" in _DEPS
    try
        @eval using CUDA
        if CUDA.functional()
            global gpu = cu
            global gpu_name = "CUDA"
        end
    catch e
        @info "CUDA load failed: $(sprint(showerror, e))"
    end
end

if gpu === nothing && "Metal" in _DEPS
    try
        @eval using Metal
        if Metal.functional()
            global gpu = mtl
            global gpu_name = "Metal"
        end
    catch e
        @info "Metal load failed: $(sprint(showerror, e))"
    end
end

if gpu !== nothing
    FT = Float32
    bc_gpu = adapt_basis(bc, gpu, FT)
    out_gpu = gpu(zeros(FT, N, N, B))
    batch_overlap!(out_gpu, bc_gpu, posA, posB)
    gpu_err = maximum(abs, out_ref .- Array(out_gpu))
    @info "2C max |Δ| (GPU=$gpu_name F32) = $gpu_err"
    @assert gpu_err < 1f-3
    print("2C KA / $gpu_name (B=$B): "); @btime batch_overlap!($out_gpu, $bc_gpu, $posA, $posB)
else
    @info "No GPU backend — skipping GPU 2C bench."
end

## --- 3-center batch ---------------------------------------------------------

B3 = 128
posA3_raw = randn(3, B3) .* 0.5
posB3_raw = randn(3, B3) .* 0.5 .+ SA[1.5, 0.0, 0.0]
posC3_raw = randn(3, B3) .* 0.5 .+ SA[0.7, 1.2, 0.3]
posA3 = posA3_raw .* u"angstrom"
posB3 = posB3_raw .* u"angstrom"
posC3 = posC3_raw .* u"angstrom"

@info "3C: B=$B3, output $(N)^3×$B3 Float64 ≈ $(round(N^3 * B3 * 8 / 2^20, digits=1)) MB"

out_ref3 = zeros(Float64, N, N, N, B3)
out_cpu3 = zeros(Float64, N, N, N, B3)

Reference.batch_V_triple_ref!(out_ref3, BS, posA3_raw, posB3_raw, posC3_raw)
batch_overlap_3c!(out_cpu3, bc, posA3, posB3, posC3)

max_err3 = maximum(abs, out_ref3 .- out_cpu3)
@info "3C max |Δ| (CPU vs Reference) = $max_err3"
@assert max_err3 < 1e-10

print("3C naive ref  (B=$B3): "); @btime Reference.batch_V_triple_ref!($out_ref3, $BS, $posA3_raw, $posB3_raw, $posC3_raw)
print("3C KA  / CPU  (B=$B3): "); @btime batch_overlap_3c!($out_cpu3, $bc, $posA3, $posB3, $posC3)

if gpu !== nothing
    FT = Float32
    out3_gpu = gpu(zeros(FT, N, N, N, B3))
    batch_overlap_3c!(out3_gpu, bc_gpu, posA3, posB3, posC3)
    gpu_err3 = maximum(abs, out_ref3 .- Array(out3_gpu))
    @info "3C max |Δ| (GPU=$gpu_name F32) = $gpu_err3"
    @assert gpu_err3 < 1f-3
    print("3C KA / $gpu_name (B=$B3): "); @btime batch_overlap_3c!($out3_gpu, $bc_gpu, $posA3, $posB3, $posC3)
end
