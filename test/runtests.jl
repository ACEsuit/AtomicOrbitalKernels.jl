using AtomicOrbitalKernels
using Test
using Random
using Unitful

include("fixtures.jl")
# detect a GPU backend once (sets `dev` / `gpu_backend`); falls back to
# `identity`, so the device tests below run on the CPU backend when no GPU is
# available. Set `TEST_BACKEND` to force a choice.
include(joinpath(@__DIR__, "utils_gpu.jl"))

# GPU kernels can't be compiled under forced bounds-checking (`--check-bounds=yes`,
# the default for `Pkg.test`): the inserted BoundsError path needs a device-side
# allocation the GPU compiler rejects. When that happens on a real GPU backend,
# skip the GPU overlap testset and remind the user afterwards (see end of file).
const _SKIP_GPU = Base.JLOptions().check_bounds == 1 && gpu_backend != "CPU"

@testset "AtomicOrbitalKernels.jl" begin
    @testset "Units" begin
        include("integrals/test_units.jl")
    end
    @testset "Reference (scalar)" begin
        include("integrals/test_reference.jl")
    end

    @testset "Atomic orbitals (eval)" begin
        include("orbitals/test_orbitals.jl")
    end

    @testset "GaussianBasis conversion + eval" begin
        include("orbitals/test_gaussianbasis.jl")
    end

    @testset "Orbital length units" begin
        include("orbitals/test_length_unit.jl")
    end

    @testset "Atomic orbitals ($gpu_backend)" begin
        include("orbitals/test_gpu.jl")
    end

    @testset "compile_basis / adapt_basis" begin
        include("integrals/test_compile.jl")
    end

    @testset "AtomicOrbitals → Cartesian compile" begin
        include("integrals/test_compile_orbitals.jl")
    end

    @testset "2-center overlap" begin
        include("integrals/test_overlap_2c.jl")
    end

    @testset "3-center overlap" begin
        include("integrals/test_overlap_3c.jl")
    end

    @testset "2C/3C overlap ($gpu_backend)" begin
        if _SKIP_GPU
            @test_skip nothing
        else
            include("integrals/test_gpu.jl")
        end
    end
end

if _SKIP_GPU
    flush(stdout)
    printstyled(stderr,
        "\n⚠ Skipped the GPU overlap tests under forced --check-bounds=yes.\n";
        color=:yellow, bold=true)
    printstyled(stderr,
        "  `Pkg.test` forces bounds-checking on, and GPU kernels can't be\n" *
        "  compiled that way. To exercise the GPU path, leave bounds-checking\n" *
        "  at its default:\n\n";
        color=:yellow)
    printstyled(stderr,
        "      using Pkg; Pkg.test(\"AtomicOrbitalKernels\"; julia_args=`--check-bounds=auto`)\n\n";
        color=:cyan, bold=true)
end
