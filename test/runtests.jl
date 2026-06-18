using AtomicOrbitalKernels
using Test
using Random
using Unitful

include("fixtures.jl")

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
    @testset "GPU (opt-in)" begin
        include("test_gpu.jl")
    end
end
