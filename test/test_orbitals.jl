using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed
import AtomicOrbitalKernels as AOK
import LuxCore
using StaticArrays, LinearAlgebra, Random, Test

@testset "construct + evaluate" begin
    rng = MersenneTwister(1234)
    h = 1e-6
    for (lbl, basis) in (("gaussian", AOK._rand_gaussian_basis()),
                         ("slater",   AOK._rand_slater_basis()))
        @testset "$lbl" begin
            Nb = length(basis)
            @test Nb > 0
            X = [ @SVector randn(3) for _ = 1:13 ]

            P = evaluate(basis, X)
            @test size(P) == (length(X), Nb)
            @test eltype(P) == Float64

            # values + derivatives are consistent
            P2, dP = evaluate_ed(basis, X)
            @test P2 ≈ P
            @test size(dP) == (length(X), Nb)
            @test eltype(eltype(dP)) == Float64    # SVector{3,Float64}

            # finite-difference check of the spatial gradient
            x0 = @SVector randn(3)
            _, dp0 = evaluate_ed(basis, [x0])
            for a = 1:3
                ea = SVector(ntuple(i -> (i == a ? 1.0 : 0.0), 3))
                Pp = evaluate(basis, [x0 + h * ea])
                Pm = evaluate(basis, [x0 - h * ea])
                fd = (Pp .- Pm) ./ (2h)
                err = maximum(abs(dp0[1, n][a] - fd[1, n]) for n = 1:Nb)
                @test err < 1e-4
            end

            # Lux-style evaluation through initialparameters/initialstates
            ps = LuxCore.initialparameters(rng, basis)
            st = LuxCore.initialstates(rng, basis)
            Plux, _ = basis(X, ps, st)
            @test size(Plux) == (length(X), Nb)
            @test Plux ≈ P     # initialised params equal the stored params

            # pullback w.r.t. parameters: directional finite-difference on Dn.ζ
            ∂P = randn(length(X), Nb)
            ∂ps = AOK.pullback_ps(∂P, basis, X, ps, st)
            ζ = ps.Dn.ζ
            V = randn(size(ζ))
            g_analytic = sum(∂ps.Dn.ζ .* V)
            lossζ(ζi) = sum(∂P .* evaluate(basis, X,
                            (Pn = ps.Pn, Dn = (ζ = ζi, D = ps.Dn.D), Ylm = ps.Ylm), st))
            g_fd = (lossζ(ζ .+ h .* V) - lossζ(ζ .- h .* V)) / (2h)
            @test isapprox(g_analytic, g_fd; rtol = 1e-4, atol = 1e-6)
        end
    end
end
