using GaussianBasisKernels: to_bohr, ang2bohr
using Unitful

@testset "to_bohr" begin
    # 1 Å -> ang2bohr Bohr
    p = reshape([1.0, 2.0, 3.0], 3, 1) .* u"angstrom"
    b = to_bohr(p, Float64)
    @test b isa Matrix{Float64}
    @test b ≈ [1.0, 2.0, 3.0] .* ang2bohr  atol = 1e-12

    # Conversion from nanometres (1 nm == 10 Å)
    p_nm = reshape([0.1, 0.1, 0.1], 3, 1) .* u"nm"
    p_a  = reshape([1.0, 1.0, 1.0], 3, 1) .* u"angstrom"
    @test to_bohr(p_nm, Float64) ≈ to_bohr(p_a, Float64)  atol = 1e-10

    # Float32 output
    p32 = reshape([1.0, 0.0, 0.0], 3, 1) .* u"angstrom"
    @test to_bohr(p32, Float32) isa Matrix{Float32}

    # Plain Real rejected with a clear message
    @test_throws ArgumentError to_bohr(randn(3, 2))
    @test_throws ArgumentError to_bohr(randn(3, 2), Float32)
end
