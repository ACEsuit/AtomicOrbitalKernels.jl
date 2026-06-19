using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed, evaluate_ref
import AtomicOrbitalKernels as AOK
import LuxCore
using StaticArrays, LinearAlgebra, Random, Test

@testset "construct + evaluate" begin
    rng = MersenneTwister(1234)
    h = 1e-6
    for (lbl, basis) in (("gaussian",   gaussian_orbitals()),
                         ("slater",     slater_orbitals()),
                         ("slater-K4",  slater_orbitals(; K = 4)))
        @testset "$lbl" begin
            Nb = length(basis)
            @test Nb > 0
            X = [ @SVector randn(3) for _ = 1:13 ]

            # KA evaluation (runs on the CPU backend here)
            P = evaluate(basis, X)
            @test size(P) == (length(X), Nb)
            @test eltype(P) == Float64

            # KA forward matches the plain reference oracle
            @test P ≈ evaluate_ref(basis, X)

            # values + derivatives consistent
            P2, dP = evaluate_ed(basis, X)
            @test P2 ≈ P
            @test size(dP) == (length(X), Nb)
            @test eltype(eltype(dP)) == Float64    # SVector{3,Float64}

            # finite-difference check of the spatial gradient
            x0 = @SVector randn(3)
            _, dp0 = evaluate_ed(basis, [x0])
            for a = 1:3
                ea = SVector(ntuple(i -> (i == a ? 1.0 : 0.0), 3))
                fd = (evaluate(basis, [x0 + h*ea]) .- evaluate(basis, [x0 - h*ea])) ./ (2h)
                @test maximum(abs(dp0[1, n][a] - fd[1, n]) for n = 1:Nb) < 1e-4
            end

            # Lux-style evaluation; the Ylm state carries the Flm matrix
            ps = LuxCore.initialparameters(rng, basis)
            st = LuxCore.initialstates(rng, basis)
            @test haskey(st.Ylm, :Flm)
            Plux, _ = basis(X, ps, st)
            @test Plux ≈ P     # initialised params equal the stored params

            # parameter pullback: directional finite-difference on ζ
            ∂P = randn(length(X), Nb)
            ∂ps = AOK.pullback_ps(∂P, basis, X, ps, st)
            ζ = ps.Rnl.ζ
            V = randn(size(ζ))
            g_analytic = sum(∂ps.Rnl.ζ .* V)
            lossζ(ζi) = sum(∂P .* evaluate(basis, X,
                            (Rnl = (ζ = ζi, D = ps.Rnl.D), Ylm = ps.Ylm), st))
            g_fd = (lossζ(ζ .+ h .* V) - lossζ(ζ .- h .* V)) / (2h)
            @test isapprox(g_analytic, g_fd; rtol = 1e-4, atol = 1e-6)
        end
    end
end
