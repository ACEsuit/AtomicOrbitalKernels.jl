# GPU backend detection for the test suite, adapted from the `restructure`
# branch of EquivariantTensors.jl.
#
# `detect_gpu_backend()` probes the *system* — no GPU package is loaded here, so
# a CPU-only machine resolves to "CPU" and installs nothing. When a GPU is
# detected, the matching backend is added to the (sandboxed) test env on demand
# and loaded. Set the `TEST_BACKEND` env var to force a choice ("CPU", "CUDA",
# "AMDGPU", "Metal", "oneAPI").
#
# Exposes:
#   `dev` (alias `gpu`)  : host<->device transfer *function* (recursive,
#                          Adapt/Functors-aware); `identity` on CPU.
#   `gpu_backend`        : chosen backend name, used as a test label.
#   `gpu_supports_f64`   : false on F32-only backends (Metal); tests run F32
#                          always and F64 only when this is true.
# The include-guard makes this safe to include from several test files.

using MLDataDevices
import Pkg

if !isdefined(Main, :___AOK_UTILS_GPU___)

    function detect_gpu_backend()
        haskey(ENV, "TEST_BACKEND") && return ENV["TEST_BACKEND"]   # manual override
        if Sys.isapple() && Sys.ARCH == :aarch64
            return "Metal"
        elseif !isnothing(Sys.which("nvidia-smi")) && success(`nvidia-smi`)
            return "CUDA"
        elseif !isnothing(Sys.which("rocm-smi")) || isdir("/dev/kfd")
            return "AMDGPU"
        elseif !isnothing(Sys.which("sycl-ls"))   # crude oneAPI probe
            return "oneAPI"
        else
            return "CPU"
        end
    end

    global gpu_backend = detect_gpu_backend()
    global gpu = global dev = identity
    global gpu_supports_f64 = true

    if gpu_backend != "CPU"
        try
            Pkg.add(gpu_backend)                  # into the sandboxed test env only
            @eval using $(Symbol(gpu_backend))
            if gpu_backend == "CUDA"
                @assert CUDA.functional();   global gpu = global dev = CUDA.cu
            elseif gpu_backend == "Metal"
                @assert Metal.functional();  global gpu = global dev = Metal.mtl
                global gpu_supports_f64 = false              # Metal is F32-only
            elseif gpu_backend == "AMDGPU"
                @assert AMDGPU.functional(); global gpu = global dev = MLDataDevices.gpu_device()
            elseif gpu_backend == "oneAPI"
                @assert oneAPI.functional(); global gpu = global dev = oneAPI.oneArray
            else
                error("unknown TEST_BACKEND = $(gpu_backend)")
            end
            @info "GPU test backend: $(gpu_backend) (F64 supported: $(gpu_supports_f64))"
        catch e
            @warn "GPU backend '$(gpu_backend)' detected but not usable; using CPU." exception=(e, catch_backtrace())
            global gpu_backend = "CPU"
            global gpu = global dev = identity
            global gpu_supports_f64 = true
        end
    end

    gpu_backend == "CPU" && @info "GPU test backend: CPU (dev = identity)."

    global ___AOK_UTILS_GPU___ = true

end
