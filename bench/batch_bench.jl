# 2C / 3C overlap benchmark.
#
# Usage:
#   julia --project=bench bench/batch_bench.jl              # CPU only
#   julia --project=bench bench/batch_bench.jl <BACKEND>    # CPU + GPU
#
# Without an argument, only the CPU benchmarks (naive `Reference` and the KA
# kernel on the CPU backend) are run.
#
# With an argument, <BACKEND> is the name of a GPU backend package (e.g.
# `CUDA`, `Metal`, `AMDGPU`, `oneAPI`) installed somewhere reachable from the
# active project's load path. The script `@eval using $BACKEND` to load it,
# then uses `MLDataDevices.gpu_device()` to obtain the corresponding device
# and `|> dev` to ship kernel inputs/outputs onto it.
#
# The bundled scalar `Reference` implementation is always run on the CPU
# (Float64) and used as the correctness oracle for both the CPU KA kernel
# (1e-10 tolerance) and the GPU kernel (1e-3 Float32 tolerance).

const RUN_GPU = !isempty(ARGS)
const BACKEND_PKG = RUN_GPU ? Symbol(ARGS[1]) : nothing

using GaussianBasisKernels
using GaussianBasisKernels: Reference
using GaussianBasis
using Molecules
using StaticArrays
using BenchmarkTools
using KernelAbstractions
using Random
using Unitful

if RUN_GPU
    using MLDataDevices
    @eval using $BACKEND_PKG
    const dev = gpu_device()
    @info "MLDataDevices device: $dev ($(typeof(dev)))"
    # Probe the device's underlying array constructor by moving a sample.
    const ArrayCtor = typeof(Float32[1.0f0] |> dev).name.wrapper
else
    @info "No GPU backend requested — running CPU benchmarks only."
end

const FT_GPU = Float32   # Metal is Float32-only; CUDA is fastest at F32.

## --- basis set --------------------------------------------------------------

const SI = Molecules.Atom(14, 28.0855, SA[0.0, 0.0, 0.0])
const BS = BasisSet("def2-SVP", [SI]; spherical=false, lib=:acsint)
const bc_cpu = compile_basis(BS)
const N = bc_cpu.nbf_total
@info "Basis: $(bc_cpu.nshells) shells, $N basis fns (cartesian), Lmax=$(bc_cpu.Lmax)"

## --- 2-center batch ---------------------------------------------------------

Random.seed!(0xC0DE)
const B = 4096
const posA_raw = randn(3, B) .* 0.5
const posB_raw = randn(3, B) .* 0.5 .+ SA[1.5, 0.0, 0.0]
const posA = posA_raw .* u"angstrom"
const posB = posB_raw .* u"angstrom"

out_ref = zeros(Float64, N, N, B)
Reference.batch_S_pair_ref!(out_ref, BS, posA_raw, posB_raw)

out_cpu = zeros(Float64, N, N, B)
batch_overlap!(out_cpu, bc_cpu, posA, posB)
@assert maximum(abs, out_ref .- out_cpu) < 1e-10 "CPU KA disagrees with Reference"

print("2C naive ref (B=$B): "); @btime Reference.batch_S_pair_ref!($out_ref, $BS, $posA_raw, $posB_raw)
print("2C KA / CPU  (B=$B): "); @btime batch_overlap!($out_cpu, $bc_cpu, $posA, $posB)

if RUN_GPU
    bc_dev = adapt_basis(bc_cpu, ArrayCtor, FT_GPU)
    out_dev = zeros(FT_GPU, N, N, B) |> dev
    batch_overlap!(out_dev, bc_dev, posA, posB)
    gpu_err = maximum(abs, out_ref .- Array(out_dev))
    @info "2C max |Δ| ($BACKEND_PKG $FT_GPU) = $gpu_err"
    @assert gpu_err < 1f-3 "GPU 2C exceeds Float32 tolerance"
    print("2C KA / $BACKEND_PKG (B=$B): "); @btime batch_overlap!($out_dev, $bc_dev, $posA, $posB)
end

## --- 3-center batch ---------------------------------------------------------

const B3 = 4096
const posA3_raw = randn(3, B3) .* 0.5
const posB3_raw = randn(3, B3) .* 0.5 .+ SA[1.5, 0.0, 0.0]
const posC3_raw = randn(3, B3) .* 0.5 .+ SA[0.7, 1.2, 0.3]
const posA3 = posA3_raw .* u"angstrom"
const posB3 = posB3_raw .* u"angstrom"
const posC3 = posC3_raw .* u"angstrom"

@info "3C: B=$B3, output $(N)^3×$B3 Float64 ≈ $(round(N^3 * B3 * 8 / 2^20, digits=1)) MB"

const RUN_NAIVE_3C = B3 <= 1024
out_cpu3 = zeros(Float64, N, N, N, B3)
batch_overlap_3c!(out_cpu3, bc_cpu, posA3, posB3, posC3)

if RUN_NAIVE_3C
    out_ref3 = zeros(Float64, N, N, N, B3)
    Reference.batch_V_triple_ref!(out_ref3, BS, posA3_raw, posB3_raw, posC3_raw)
    @assert maximum(abs, out_ref3 .- out_cpu3) < 1e-10 "CPU KA 3C disagrees with Reference"
    print("3C naive ref (B=$B3): "); @btime Reference.batch_V_triple_ref!($out_ref3, $BS, $posA3_raw, $posB3_raw, $posC3_raw)
else
    @info "B3=$B3 > 1024 — skipping naive 3C reference (too slow); CPU KA used as the oracle."
end
print("3C KA / CPU  (B=$B3): "); @btime batch_overlap_3c!($out_cpu3, $bc_cpu, $posA3, $posB3, $posC3)

if RUN_GPU
    out_dev3 = zeros(FT_GPU, N, N, N, B3) |> dev
    batch_overlap_3c!(out_dev3, bc_dev, posA3, posB3, posC3)
    oracle3 = RUN_NAIVE_3C ? out_ref3 : out_cpu3
    gpu_err3 = maximum(abs, oracle3 .- Array(out_dev3))
    @info "3C max |Δ| ($BACKEND_PKG $FT_GPU vs $(RUN_NAIVE_3C ? "ref" : "CPU KA")) = $gpu_err3"
    @assert gpu_err3 < 1f-3 "GPU 3C exceeds Float32 tolerance"
    print("3C KA / $BACKEND_PKG (B=$B3): "); @btime batch_overlap_3c!($out_dev3, $bc_dev, $posA3, $posB3, $posC3)
end
