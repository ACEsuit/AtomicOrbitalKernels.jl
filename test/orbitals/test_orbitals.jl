using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed, evaluate_ref
import AtomicOrbitalKernels as AOK
import LuxCore
using StaticArrays, LinearAlgebra, Random, Test
using ForwardDiff: Dual, value, partials
using DecoratedParticles: PState, VState

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
            dY_ad = [ partials(Yd[i, n], 1) for i = 1:length(X), n = 1:Nb ]
            dPdU = [ dot(dP[i, n], U[i]) for i = 1:length(X), n = 1:Nb ]
            @test dY_ad ≈ dPdU

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

(lbl, basis) = ("gaussian", gaussian_orbitals(4, 3; nspecies = 3, zlist = zlist))
@testset "multi-species (PState input)" begin
    rng = MersenneTwister(4321)
    zlist = (6, 1, 8)          # species labels distinct from their 1:NZ indices
    for (lbl, basis) in (("gaussian",
                            gaussian_orbitals(4, 3; nspecies = 3, zlist = zlist)),
                         ("slater-K4",
                            slater_orbitals(4, 3; K = 4, nspecies = 3, zlist = zlist)))
        @testset "$lbl" begin
            Nb = length(basis)
            @test AOK.nspecies(basis.Rnl) == 3
            nX = 17
            X = [ PState(𝐫 = (@SVector randn(3)), S = rand(rng, zlist)) for _ = 1:nX ]

            # KA forward (via the DecoratedParticles input layer) ≈ reference
            P = evaluate(basis, X)
            @test size(P) == (nX, Nb)
            @test P ≈ evaluate_ref(basis, X)

            P2, dP = evaluate_ed(basis, X)
            @test P2 ≈ P
            @test size(dP) == (nX, Nb)
            @test eltype(dP) <: VState

            # ForwardDiff spatial gradient (species fixed per point)
            U  = [ @SVector randn(3) for _ = 1:nX ]
            Xd = [ PState(𝐫 = X[i].𝐫 + Dual(0.0, 1.0) * U[i], S = X[i].S) for i = 1:nX ]
            Yd = evaluate_ref(basis, Xd)
            @test value.(Yd) ≈ P
            dY_ad = [ partials(Yd[i, n], 1) for i = 1:nX, n = 1:Nb ]
            dPdU = [ dot(dP[i, n].𝐫, U[i]) for i = 1:nX, n = 1:Nb ]
            @test dY_ad ≈ dPdU

            # Lux-style params: ζ/D carry the species axis as a plain Array
            ps = LuxCore.initialparameters(rng, basis)
            st = LuxCore.initialstates(rng, basis)
            @test ps.Rnl.ζ isa Array && size(ps.Rnl.ζ, 3) == 3
            Plux, _ = basis(X, ps, st)
            @test Plux ≈ P

            # parameter pullback (ζ, D) vs ForwardDiff directional derivatives.
            # The 3D perturbation covers every species slice; each point feeds
            # only its own species, so this is exactly the per-species pullback.
            ∂P = randn(nX, Nb)
            ∂ps = AOK.pullback_ps(∂P, basis, X, ps, st)
            @test size(∂ps.Rnl.ζ) == size(ps.Rnl.ζ)
            ζ = ps.Rnl.ζ;  D = ps.Rnl.D
            V = randn(size(ζ));  W = randn(size(D))
            lossζ(ζi) = sum(∂P .* evaluate(basis, X,
                            (Rnl = (ζ = ζi, D = D), Ylm = ps.Ylm), st))
            lossD(Di) = sum(∂P .* evaluate(basis, X,
                            (Rnl = (ζ = ζ, D = Di), Ylm = ps.Ylm), st))
            @test sum(∂ps.Rnl.ζ .* V) ≈ partials(lossζ(ζ .+ Dual(0.0,1.0).*V), 1)
            @test sum(∂ps.Rnl.D .* W) ≈ partials(lossD(D .+ Dual(0.0,1.0).*W), 1)

            # rrule merges the X-pullback (via evaluate_ed) and the param pullback;
            # a PState input yields a VState X-cotangent (position grad in `𝐫`)
            y, pb = AOK.rrule(evaluate, basis, X, ps, st)
            @test y ≈ P
            _, _, ∂X, ∂ps_r, _ = pb(∂P)
            @test eltype(∂X) <: VState
            @test ∂ps_r.Rnl.ζ ≈ ∂ps.Rnl.ζ
            @test ∂ps_r.Rnl.D ≈ ∂ps.Rnl.D
            @test sum(∂P .* partials.(Yd, 1)) ≈
                  sum(dot(∂X[j].𝐫, U[j]) for j in eachindex(X))

            # the radial basis also accepts PState directly: exercise `_radii`,
            # the radial-layer PState overloads, and the 3-arg species reference
            sidxX = AOK._species_indices(basis.Rnl, X)
            Rr = evaluate(basis.Rnl, X, ps.Rnl, st.Rnl)
            @test Rr ≈ evaluate(basis.Rnl, AOK._radii(X), sidxX, ps.Rnl, st.Rnl)
            @test Rr ≈ evaluate_ref(basis.Rnl, AOK._radii(X), sidxX)
            Rr2, _ = evaluate_ed(basis.Rnl, X, ps.Rnl, st.Rnl)
            @test Rr2 ≈ Rr
            ∂Rr = randn(nX, length(basis.Rnl))
            @test AOK.pullback_ps(∂Rr, basis.Rnl, X, ps.Rnl, st.Rnl).ζ ≈
                  AOK.pullback_ps(∂Rr, basis.Rnl, AOK._radii(X), sidxX, ps.Rnl, st.Rnl).ζ
        end
    end

    # a species absent from the batch contributes zero parameter gradient
    basis = gaussian_orbitals(4, 3; nspecies = 3, zlist = zlist)
    Xsub = [ PState(𝐫 = (@SVector randn(3)), S = rand(rng, (6, 1))) for _ = 1:12 ]
    ps = AOK._static_params(basis);  st = AOK._static_state(basis)
    ∂ps = AOK.pullback_ps(randn(12, length(basis)), basis, Xsub, ps, st)
    σ8 = AOK._z2i(basis.Rnl, 8)
    @test all(iszero, ∂ps.Rnl.ζ[:, :, σ8])
    @test all(iszero, ∂ps.Rnl.D[:, :, σ8])

    # unknown species → clean error (the only error path the feature adds)
    @test_throws ErrorException AOK._z2i(basis.Rnl, 99)
    @test_throws ErrorException evaluate(basis, [PState(𝐫 = (@SVector randn(3)), S = 99)])

    # positions-only call defaults to species 1 (= zlist[1]) on a multi-species basis
    Xdef = [ @SVector randn(3) for _ = 1:9 ]
    @test evaluate(basis, Xdef) ≈
          evaluate(basis, [PState(𝐫 = x, S = zlist[1]) for x in Xdef])

    # regression: a single-species basis evaluated via PState matches positions
    b1 = gaussian_orbitals(3, 2)
    Xpos = [ @SVector randn(3) for _ = 1:11 ]
    Xps  = [ PState(𝐫 = x, S = 1) for x in Xpos ]
    @test evaluate(b1, Xps) ≈ evaluate(b1, Xpos)
end
