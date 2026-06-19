# GPU backend detection for the test suite, adapted from EquivariantTensors.jl.
#
# Exposes `dev` (alias `gpu`), the device-transfer / array constructor, and a
# `dev_name` label. When no functional GPU is found, `dev = identity`, so the
# GPU test code runs unchanged on the CPU backend (in Float32) instead of being
# skipped. The include-guard makes this safe to include from several test
# files; only the first include runs the detection.

if !isdefined(Main, :___AOK_UTILS_GPU___)

    global dev = global gpu = identity
    global dev_name = "CPU"
    global __has_gpu = false

    try
        using CUDA
        if CUDA.functional()
            @info "Tests: using CUDA"
            global dev = global gpu = cu
            global dev_name = "CUDA"
            global __has_gpu = true
        else
            @info "Tests: CUDA present but not functional"
        end
    catch
        @info "Tests: CUDA not available"
    end

    if !__has_gpu
        try
            using Metal
            if Metal.functional()
                @info "Tests: using Metal"
                global dev = global gpu = mtl
                global dev_name = "Metal"
                global __has_gpu = true
            else
                @info "Tests: Metal present but not functional"
            end
        catch
            @info "Tests: Metal not available"
        end
    end

    __has_gpu ||
        @info "Tests: no functional GPU — running device tests on the CPU backend"

    global ___AOK_UTILS_GPU___ = true
end
