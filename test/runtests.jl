using AtomicOrbitalKernels
using Test
using Random
using Unitful

include("fixtures.jl")
# detect a GPU backend once (sets `dev` / `dev_name`); falls back to `identity`,
# so the device tests below run on the CPU backend when no GPU is available.
include(joinpath(@__DIR__, "utils_gpu.jl"))

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
    @testset "2C/3C overlap ($dev_name)" begin
        include("test_gpu.jl")
    end
    @testset "Atomic orbitals ($dev_name)" begin
        include("orbitals/test_gpu.jl")
    end
end
