using AtomicOrbitalKernels
using Test
using Random
using Unitful
using Pkg

include("fixtures.jl")

# detect an available GPU backend (CUDA or Metal) once, shared by the opt-in GPU
# test files. `_gpu === nothing` means no functional backend, so those tests are
# skipped.
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
_gpu === nothing && @info "No GPU backend available — skipping GPU tests."

@testset "AtomicOrbitalKernels.jl" begin
    @testset "Units" begin
        include("test_units.jl")
    end
    @testset "Reference (scalar)" begin
        include("test_reference.jl")
    end
    @testset "compile_basis / adapt_basis" begin
        include("test_compile.jl")
    end
    @testset "2-center overlap" begin
        include("test_overlap_2c.jl")
    end
    @testset "3-center overlap" begin
        include("test_overlap_3c.jl")
    end
    @testset "Atomic orbitals (eval)" begin
        include("orbitals/test_orbitals.jl")
    end
    @testset "2C/3C overlap (GPU, opt-in)" begin
        include("test_gpu.jl")
    end
    @testset "Atomic orbitals (GPU, opt-in)" begin
        include("orbitals/test_gpu.jl")
    end
end
