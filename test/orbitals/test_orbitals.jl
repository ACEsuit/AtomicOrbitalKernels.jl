using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed, evaluate_ref
import AtomicOrbitalKernels as AOK
import LuxCore
using StaticArrays, LinearAlgebra, Random, Test
using ForwardDiff: Dual, value, partials

@testset "construct + evaluate" begin
    rng = MersenneTwister(1234)
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

            # parameter pullback (ζ and D), checked against ForwardDiff
            # directional derivatives (same quantity as the analytic pullback)
            ∂P = randn(length(X), Nb)
            ∂ps = AOK.pullback_ps(∂P, basis, X, ps, st)
            ζ = ps.Rnl.ζ;  D = ps.Rnl.D
            V = randn(size(ζ));  W = randn(size(D))
            lossζ(ζi) = sum(∂P .* evaluate(basis, X,
                            (Rnl = (ζ = ζi, D = D), Ylm = ps.Ylm), st))
            lossD(Di) = sum(∂P .* evaluate(basis, X,
                            (Rnl = (ζ = ζ, D = Di), Ylm = ps.Ylm), st))
            @test sum(∂ps.Rnl.ζ .* V) ≈ partials(lossζ(ζ .+ Dual(0.0,1.0).*V), 1)
            @test sum(∂ps.Rnl.D .* W) ≈ partials(lossD(D .+ Dual(0.0,1.0).*W), 1)

            # rrule merges the X-pullback (via evaluate_ed) and the param pullback
            y, pb = AOK.rrule(evaluate, basis, X, ps, st)
            @test y ≈ P
            _, _, ∂X, ∂ps_r, _ = pb(∂P)
            @test ∂ps_r.Rnl.ζ ≈ ∂ps.Rnl.ζ
            @test ∂ps_r.Rnl.D ≈ ∂ps.Rnl.D
            # vjp identity: ∑ ∂P .* d/dε ϕ(X+εU) == ∑ dot(∂X[j], U[j])  (Yd, U above)
            @test sum(∂P .* partials.(Yd, 1)) ≈
                  sum(dot(∂X[j], U[j]) for j in eachindex(X))
        end
    end
end
