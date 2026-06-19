using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed, evaluate_ref
import AtomicOrbitalKernels as AOK
import LuxCore
using StaticArrays, LinearAlgebra, Random, Test
using ForwardDiff: Dual, value, partials

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

            # ForwardDiff check of the spatial gradient: seed each point with a
            # random direction `U`, so the dual partial of `evaluate_ref` is the
            # directional derivative `dot(∇ϕ, U)`. Machine precision, and at the
            # same time this checks that the reference accepts Dual inputs.
            U  = [ @SVector randn(3) for _ = 1:length(X) ]
            Xd = [ X[i] + Dual(0.0, 1.0) * U[i] for i in eachindex(X) ]
            Yd = evaluate_ref(basis, Xd)
            @test value.(Yd) ≈ P
            @test all(isapprox(partials(Yd[i, n], 1), dot(dP[i, n], U[i]);
                               atol = 1e-10, rtol = 1e-9)
                      for i = 1:length(X), n = 1:Nb)

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
