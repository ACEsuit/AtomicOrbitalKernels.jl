# Learn GTO radial parameters (ζ, D) from 2-center overlaps.
#
# A reference STO-3G carbon basis generates 2c overlap matrices for random
# C-C pairs. A second basis starts from ~10% perturbed parameters and is
# fitted by minimizing the LSQ loss over the training overlaps with BFGS,
# using the analytic parameter pullback `batch_overlap_pb_ps`. Train and
# test RMSE are reported before and after optimization.
#
# Run:  julia --project=examples/learnbasis examples/learnbasis/learnbasis.jl

using AtomicOrbitalKernels
import AtomicOrbitalKernels as AOK
using GaussianBasis: BasisSet
using AtomsBase: ChemicalSpecies
using DecoratedParticles: PState
import LuxCore
using Optim
using StaticArrays, LinearAlgebra, Random, Printf

# ---------------------------------------------------------------- config

basisname = "sto-3g"        # all-K=3 contractions: every D entry matters
rvdw      = 1.7             # vdW radius of C in Å
rrange    = (0.5, 4.0)      # pair distances in units of rvdw
B_train   = 1000
B_test    = 1000
pert      = 0.1             # ~10% relative parameter perturbation

# ------------------------------------------------------- data generation

# Random C-C pairs: distance r ~ U(rrange)*rvdw, uniform random direction,
# random center (overlaps are translation invariant; randomizing centers
# keeps the data generic).
function make_pairs(B, rng)
    sp = ChemicalSpecies(6)
    XA = Vector{PState}(undef, B); XB = Vector{PState}(undef, B)
    for i in 1:B
        r = (rrange[1] + (rrange[2] - rrange[1]) * rand(rng)) * rvdw
        u = normalize(randn(rng, SVector{3, Float64}))
        c = randn(rng, SVector{3, Float64})
        XA[i] = PState(𝐫 = c, S = sp)
        XB[i] = PState(𝐫 = c + r * u, S = sp)
    end
    return [XA...], [XB...]   # narrow eltype for the kernels
end

rng = MersenneTwister(1234)
XAtr, XBtr = make_pairs(B_train, rng)
XAte, XBte = make_pairs(B_test, rng)

# ------------------------------------------- reference basis & overlaps

orb = gaussian_orbitals(BasisSet(basisname, "C 0.0 0.0 0.0"); length_unit = :Å)
ps_ref = LuxCore.initialparameters(rng, orb)   # the actual STO-3G (ζ, D)
st     = LuxCore.initialstates(rng, orb)

S_ref_tr = batch_overlap(orb, XAtr, XBtr, ps_ref, st)
S_ref_te = batch_overlap(orb, XAte, XBte, ps_ref, st)

# ------------------------------------------------------ perturbed start

ζ_ref = ps_ref.Rnl.ζ;  D_ref = ps_ref.Rnl.D
ζ0 = ζ_ref .* exp.(pert .* randn(rng, size(ζ_ref)))   # multiplicative: ζ > 0
D0 = D_ref .* (1 .+ pert .* randn(rng, size(D_ref)))

# ------------------------------------------------- LSQ loss & optimizer

# Optimization variable θ = [vec(log.(ζ)); vec(D)]: log-parameterizing ζ
# keeps it positive, so line searches cannot step into ζ ≤ 0 blowups.
nζ = length(ζ_ref)
θ2ps(θ) = (Rnl = (ζ = exp.(reshape(θ[1:nζ], size(ζ_ref))),
                  D = reshape(θ[nζ+1:end], size(D_ref))),
           Ylm = ps_ref.Ylm)

# L(θ) = 1/(2B) Σ_b ‖S_b(θ) - S_ref_b‖²_F, gradient via a single VJP with
# cotangent ∂S = (S - S_ref)/B; ∂logζ = ζ .* ∂ζ (chain rule).
function fg!(F, G, θ)
    ps = θ2ps(θ)
    S = batch_overlap(orb, XAtr, XBtr, ps, st)
    ∂S = (S .- S_ref_tr) ./ B_train
    if G !== nothing
        ∂ps = AOK.batch_overlap_pb_ps(∂S, orb, XAtr, XBtr, ps, st)
        G[1:nζ]     .= vec(ps.Rnl.ζ .* ∂ps.Rnl.ζ)
        G[nζ+1:end] .= vec(∂ps.Rnl.D)
    end
    if F !== nothing
        return 0.5 * B_train * sum(abs2, ∂S)
    end
    return nothing
end

θ0 = [vec(log.(ζ0)); vec(D0)]
res = optimize(Optim.NLSolversBase.only_fg!(fg!), θ0, BFGS(),
               Optim.Options(g_tol = 1e-10, show_trace = true,
                             iterations = 500))
display(res)
ps_opt = θ2ps(Optim.minimizer(res))

# -------------------------------------------------------------- report

rmse(ps, XA, XB, S_ref) =
    sqrt(sum(abs2, batch_overlap(orb, XA, XB, ps, st) .- S_ref) / length(S_ref))

ps0 = θ2ps(θ0)
@printf("\n%-12s %14s %14s\n", "", "train RMSE", "test RMSE")
@printf("%-12s %14.6e %14.6e\n", "perturbed",
        rmse(ps0, XAtr, XBtr, S_ref_tr), rmse(ps0, XAte, XBte, S_ref_te))
@printf("%-12s %14.6e %14.6e\n", "optimized",
        rmse(ps_opt, XAtr, XBtr, S_ref_tr), rmse(ps_opt, XAte, XBte, S_ref_te))

# Parameter recovery, per radial. Two caveats: (a) the overlap normalizes each
# contracted radial, so D is identifiable only up to a per-radial scaling —
# compare D_n/‖D_n‖ (sign-aligned); (b) the tight core primitives (1s: ζ ≈ 71,
# 13, 3.5) barely contribute to overlaps at these distances, so those parameters
# are only weakly identifiable — the overlaps match without recovering them.
ζ_opt = ps_opt.Rnl.ζ;  D_opt = ps_opt.Rnl.D
@printf("\nparameter recovery per radial (D up to per-radial scale):\n")
@printf("%-10s %14s %14s\n", "radial", "rel.err ζ", "err D")
for i in 1:size(ζ_ref, 1)
    ζ_err = maximum(abs.(ζ_opt[i, :, :] .- ζ_ref[i, :, :])
                    ./ abs.(ζ_ref[i, :, :]))
    d1 = normalize(vec(D_opt[i, :, :]));  d2 = normalize(vec(D_ref[i, :, :]))
    D_err = min(norm(d1 - d2), norm(d1 + d2))
    nl = orb.Rnl.spec[i]
    @printf("(n=%d, l=%d) %14.3e %14.3e\n", nl.n, nl.l, ζ_err, D_err)
end
