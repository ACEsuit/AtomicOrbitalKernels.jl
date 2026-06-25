# Validate the AtomicOrbitals evaluation against Gaussian basis sets loaded with
# GaussianBasis.jl. GaussianBasis has no AO-value evaluator (it only computes
# integrals), so we (1) convert each `BasisSet` to our `AtomicOrbitals` and compare
# our `evaluate` to an independent reference built from the same shell data, and
# (2) cross-check the orbital overlap against GaussianBasis's own `overlap` (libcint).
#
# Key facts that make this clean (see the test for the verification):
#  - GaussianBasis spherical `coef` are L2-normalized and SpheriCart `SolidHarmonics`
#    default to `:L2`, so feeding `(ζ = exp, D = coef, poly = 0)` reproduces the
#    GaussianBasis-defined AO with NO extra normalization constant.
#  - The `r^l` lives in the solid harmonic, so the radial power is 0 (GTO).
#  - We use single atoms at the ORIGIN: our orbitals are centre-free, GaussianBasis
#    AOs are atom-centred, so they coincide only there.
#  - libcint and SpheriCart order the `m` within a shell differently — value
#    comparisons are done per-`(l,m)` (order-safe) and the overlap cross-check uses
#    order-independent sorted eigenvalues.

using AtomicOrbitalKernels
import AtomicOrbitalKernels as AOK
using AtomicOrbitalKernels: evaluate, evaluate_ed, evaluate_ref
using GaussianBasis: BasisSet, overlap
using StaticArrays, LinearAlgebra, Test
import SpheriCart

const _NTNL  = NamedTuple{(:n, :l), Tuple{Int, Int}}
const _NTNNL = NamedTuple{(:n1, :n2, :l), Tuple{Int, Int, Int}}

# convert a (spherical) GaussianBasis.BasisSet to our AtomicOrbitals: one radial per
# shell, `ζ = shell.exp`, `D = shell.coef` (already normalized — taken verbatim),
# `poly = 0`. Single species; atoms assumed centred at the origin.
function _orbitals_from_basisset(bs; T = Float64)
    shells = bs.basis
    nRad = length(shells)
    K = maximum(length(s.exp) for s in shells)
    ζ = ones(T, nRad, K, 1)
    D = zeros(T, nRad, K, 1)
    spec = _NTNL[];  nnspec = _NTNNL[];  ncount = Dict{Int, Int}()
    for (k, s) in enumerate(shells)
        nk = length(s.exp)
        @views ζ[k, 1:nk, 1] .= s.exp        # remaining ζ stay 1 (inert; D = 0 there)
        @views D[k, 1:nk, 1] .= s.coef
        n = (ncount[s.l] = get(ncount, s.l, 0) + 1)
        push!(spec,   (n = n, l = s.l))
        push!(nnspec, (n1 = n, n2 = 1, l = s.l))
    end
    Lmax = maximum(s.l for s in shells)
    radial = AOK.GaussianTypeRadials(ζ, D, spec, nnspec, (1,))
    return AOK.AtomicOrbitals(radial, SpheriCart.SolidHarmonics(Lmax))
end

# independent reference: χ_{l,m}(R) = [Σ_k coef_k exp(-exp_k |R|²)] · Z_{l,m}(R),
# with Z the SpheriCart solid harmonic at linear index l²+l+m+1. Assembled in our
# orbital column order (shells in `bs.basis` order, m = -l:l), matching `AtomicOrbitals`.
function _gb_reference(bs, Xpos)
    shells = bs.basis
    Lmax = maximum(s.l for s in shells)
    sh = SpheriCart.SolidHarmonics(Lmax)
    P = zeros(eltype(eltype(Xpos)), length(Xpos), bs.nbas)
    col = 0
    for s in shells
        for (i, R) in enumerate(Xpos)
            rad = sum(s.coef[k] * exp(-s.exp[k] * dot(R, R)) for k in eachindex(s.exp))
            Z = SpheriCart.compute(sh, R)
            for (j, m) in enumerate(-s.l:s.l)
                P[i, col + j] = rad * Z[s.l^2 + s.l + m + 1]
            end
        end
        col += 2 * s.l + 1
    end
    return P
end

# analytic overlap of the converted orbitals (single atom at the origin): different
# (l,m) are orthogonal (L2 harmonics), same (l,m) couple radially via the Gaussian
# moment ∫₀^∞ exp(-γr²) r^{2l+2} dr = ½ Γ(l+3/2) γ^{-(l+3/2)}, Γ(l+3/2) =
# (2l+1)!!/2^{l+1} · √π. Used as a grid-free, libcint-independent overlap oracle.
function _analytic_overlap(orb)
    ζ = orb.Rnl.ζ;  D = orb.Rnl.D                      # [nRad × K × 1] MArrays
    spec = orb.spec                                    # (n,l,m) per orbital
    nb = length(orb)
    S = zeros(Float64, nb, nb)
    dfac(n) = (p = 1.0; for k = 1:2:n; p *= k; end; p) # double factorial n!!
    for i = 1:nb, j = 1:nb
        (spec[i].l == spec[j].l && spec[i].m == spec[j].m) || continue
        l = spec[i].l
        ki = orb.radidx[i];  kj = orb.radidx[j]
        g = 0.5 * dfac(2l + 1) / 2.0^(l + 1) * sqrt(pi)   # ½ Γ(l+3/2)
        acc = 0.0
        for a = 1:size(ζ, 2), b = 1:size(ζ, 2)
            acc += D[ki, a, 1] * D[kj, b, 1] *
                   (ζ[ki, a, 1] + ζ[kj, b, 1])^(-(l + 1.5))
        end
        S[i, j] = g * acc
    end
    return S
end

@testset "GaussianBasis conversion + evaluation" begin
    # spherical basis sets (default); single atom at the origin. sto-3g/H is the
    # minimal smoke; 6-31g/C adds contracted s,p; cc-pvdz/O and def2-svp/O add d.
    for (name, el) in (("sto-3g", "H"), ("6-31g", "C"),
                       ("cc-pvdz", "O"), ("def2-svp", "O"))
        @testset "$name / $el" begin
            bs = BasisSet(name, "$el 0.0 0.0 0.0")
            orb = _orbitals_from_basisset(bs)

            # structural: one (2l+1)-fold orbital block per shell
            @test length(orb) == bs.nbas
            @test length(orb) == sum(2 * s.l + 1 for s in bs.basis)

            # --- A. values match the GaussianBasis-defined AO (machine precision) ---
            X = [ @SVector randn(3) for _ = 1:12 ]
            ref = _gb_reference(bs, X)
            P = evaluate(orb, X)
            @test all(isfinite, P)
            @test P ≈ ref atol = 1e-10
            @test evaluate_ref(orb, X) ≈ ref atol = 1e-10
            Ped, _ = evaluate_ed(orb, X)
            @test Ped ≈ ref atol = 1e-10

            # Float32 conversion + evaluation path
            orb32 = _orbitals_from_basisset(bs; T = Float32)
            X32 = [ SVector{3, Float32}(x) for x in X ]
            P32 = evaluate(orb32, X32)
            @test eltype(P32) == Float32
            @test P32 ≈ ref atol = 1e-4

            # --- B. overlap cross-check against GaussianBasis's libcint `overlap` ---
            # both reflect the same (possibly non-unit, e.g. def2-svp) normalization;
            # sorted eigenvalues are independent of the libcint-vs-SpheriCart m-order.
            ev_gb = sort(eigvals(Symmetric(overlap(bs))))
            ev_an = sort(eigvals(Symmetric(_analytic_overlap(orb))))
            @test ev_an ≈ ev_gb rtol = 1e-6
        end
    end
end
