using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed, load_basis, bundled_basis_names
import AtomicOrbitalKernels as AOK
using StaticArrays, Test

@testset "basis-set parameter reading" begin
    @test "sto-3g" in bundled_basis_names()

    # STO-3G hydrogen: one s shell, 3 primitives, known exponents/coefficients
    pbH = load_basis("sto-3g", 1)
    @test length(pbH.shells) == 1
    s = pbH.shells[1]
    @test s.l == 0
    @test s.exponents ≈ [3.42525091, 0.62391373, 0.16885540] rtol = 1e-6
    @test s.coefficients ≈ [0.15432897, 0.53532814, 0.44463454] rtol = 1e-6

    # STO-3G carbon: the sp shell splits into s + p → radials (1s, 2s, 2p),
    # i.e. 5 orbitals (1s, 2s, 2p×3)
    pbC = load_basis("sto-3g", 6)
    @test length(pbC.shells) == 3
    @test sort([sh.l for sh in pbC.shells]) == [0, 0, 1]
    basisC = gaussian_orbitals(pbC)
    @test length(basisC) == 5

    # build + evaluate across several sets / elements (varying contraction K)
    for (name, Z) in (("sto-3g", 1), ("6-31g", 6), ("def2-svp", 8), ("cc-pvdz", 6))
        b = gaussian_orbitals(name, Z)
        pb = load_basis(name, Z)
        @test length(b) == sum(2 * sh.l + 1 for sh in pb.shells)   # Σ (2l+1)
        X = [ @SVector randn(3) for _ = 1:7 ]
        P = evaluate(b, X)
        @test size(P) == (length(X), length(b))
        @test all(isfinite, P)
        P2, dP = evaluate_ed(b, X)
        @test P2 ≈ P
        @test all(v -> all(isfinite, v), dP)
    end

    # T = Float32 path
    b32 = gaussian_orbitals("sto-3g", 1; T = Float32)
    @test eltype(b32.Rnl.ζ) == Float32

    @test_throws ErrorException load_basis("not-a-basis", 1)
    @test_throws ErrorException load_basis("sto-3g", 92)   # U not in H–Ar bundle
end
