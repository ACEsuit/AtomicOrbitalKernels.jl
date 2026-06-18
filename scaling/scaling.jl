# Scaling sweep for the 2C / 3C overlap kernels.
#
# Before running this benchmark ensure that the development version of 
# AtomicOrbitalKernels is added to the active project (e.g. `] add ..`)
#
# Usage:
#   julia --project=scaling -t auto scaling/scaling.jl              # CPU (Float64)
#   julia --project=scaling -t auto scaling/scaling.jl CPU          # explicit CPU
#   julia --project=scaling scaling/scaling.jl <BACKEND>            # GPU (Float32)
#
# `<BACKEND>` is the name of a GPU backend package installed somewhere on the
# active project's load path (e.g. `CUDA`, `Metal`, `AMDGPU`, `oneAPI`). The
# script `@eval using $BACKEND`s it, then asks `MLDataDevices.gpu_device()` for
# the matching device.
#
# CPU runs parallel across Julia threads, so the CPU column depends on
# `--threads` (`-t`). Thread count is irrelevant once a GPU backend is selected.
#

const BACKEND = (isempty(ARGS) || ARGS[1] == "CPU") ? "CPU" : ARGS[1]
const USE_GPU = BACKEND != "CPU"

using AtomicOrbitalKernels
using GaussianBasis
using Molecules
using StaticArrays
using BenchmarkTools
using KernelAbstractions
using Random
using Unitful
using Printf

if USE_GPU && BACKEND == "CUDA" 
    using CUDA 
    const dev = cu 
    const ArrayCtor = typeof(Float32[1.0f0] |> dev).name.wrapper
    const FT = Float32
    move(x) = x |> dev
elseif USE_GPU
    using MLDataDevices
    @eval using $(Symbol(BACKEND))
    const dev = gpu_device()
    const ArrayCtor = typeof(Float32[1.0f0] |> dev).name.wrapper
    const FT = Float32
    move(x) = x |> dev
else
    const ArrayCtor = Array
    const FT = Float64
    move(x) = x
end 

## --- basis --------------------------------------------------------------

const SI = Molecules.Atom(14, 28.0855, SA[0.0, 0.0, 0.0])
const BS = BasisSet("def2-SVP", [SI]; spherical=false, lib=:acsint)
const bc_host = compile_basis(BS)
const bc = USE_GPU ? adapt_basis(bc_host, ArrayCtor, FT) : bc_host
const N = bc_host.nbf_total

threads_str = USE_GPU ? "(GPU)" : "$(Threads.nthreads()) thread(s)"
@info "backend=$BACKEND  basis=$(bc_host.nshells) shells, $N basis fns, Lmax=$(bc_host.Lmax)  FT=$FT  $threads_str"

## --- helpers ------------------------------------------------------------

function _random_positions(B; offset = (0.0, 0.0, 0.0), scale = 0.5)
    raw = randn(3, B) .* scale
    raw[1, :] .+= offset[1]; raw[2, :] .+= offset[2]; raw[3, :] .+= offset[3]
    return raw .* u"angstrom"
end

# Format an elapsed time (seconds) into a short human string.
function _fmt(t)
    isnan(t) && return "      —"
    t < 1e-3  && return @sprintf("%6.2f μs", t * 1e6)
    t < 1.0   && return @sprintf("%6.2f ms", t * 1e3)
    return                @sprintf("%6.2f s ", t)
end

function _time_2c(B)
    posA = _random_positions(B)
    posB = _random_positions(B; offset = (1.5, 0.0, 0.0))
    out  = move(zeros(FT, N, N, B))
    batch_overlap!(out, bc, posA, posB)           # warm-up
    return @belapsed batch_overlap!($out, $bc, $posA, $posB)
end

function _time_3c(B)
    posA = _random_positions(B)
    posB = _random_positions(B; offset = (1.5, 0.0, 0.0))
    posC = _random_positions(B; offset = (0.7, 1.2, 0.3))
    out  = move(zeros(FT, N, N, N, B))
    batch_overlap_3c!(out, bc, posA, posB, posC)  # warm-up
    return @belapsed batch_overlap_3c!($out, $bc, $posA, $posB, $posC)
end

## --- sweep + table ------------------------------------------------------

const BS_LIST = [2^k for k in 7:14]   # 128, 256, …, 16384

Random.seed!(0xC0DE)

println()
println("| batch size |         C2 |         C3 |")
println("|-----------:|-----------:|-----------:|")
for B in BS_LIST
    t2 = try _time_2c(B) catch e; @warn "C2 B=$B failed: $(sprint(showerror, e))"; NaN end
    GC.gc()
    t3 = try _time_3c(B) catch e; @warn "C3 B=$B failed: $(sprint(showerror, e))"; NaN end
    GC.gc()
    @printf("| %10d | %s | %s |\n", B, _fmt(t2), _fmt(t3))
end
