# Validate `gaussian_orbitals(::GaussianBasis.BasisSet)` against Gaussian basis sets
# loaded with GaussianBasis.jl. GaussianBasis has no AO-value evaluator (it only
# computes integrals), so we (1) compare our `evaluate` to an independent reference
# built from the same shell data, and (2) cross-check the single-atom orbital overlap
# against GaussianBasis's own `overlap` (libcint).
#
# The converter maps each chemical ELEMENT to one species (per-species `(ζ,D)`, shared
# `(n,l)` spec, zero-padded where an element lacks a shell). `coef` are taken verbatim
# (L2-normalized) and SpheriCart `SolidHarmonics` are `:L2`, so AO values match with no
# normalization constant; `poly = 0` (the `r^l` is in the solid harmonic). Orbitals are
# centre-free → single atoms at the origin; libcint vs SpheriCart order `m` differently,
# so value checks are per-`(l,m)` and the overlap check uses sorted eigenvalues.

using AtomicOrbitalKernels
import AtomicOrbitalKernels as AOK
using AtomicOrbitalKernels: evaluate, evaluate_ed, evaluate_ref
using GaussianBasis: BasisSet, overlap
using AtomsBase: ChemicalSpecies, atomic_number
using DecoratedParticles: PState
using StaticArrays, LinearAlgebra, Test
import SpheriCart

# independent reference, keyed by per-point atomic number `Zs[i]`: for orbital
# `(n,l,m)` use that element's n-th shell of angular momentum `l` (0 if it lacks one,
# i.e. a padded slot), χ = [Σ_k coef_k exp(-exp_k |R|²)] · Z_{l,m}(R). Column order
# matches `orb.spec` (shells per element are angular-momentum-ordered in GaussianBasis).
function _gb_ref(orb, bs, pos, Zs)
    spec = orb.spec
    Lmax = maximum(s.l for s in bs.basis)
    sh = SpheriCart.SolidHarmonics(Lmax)
    P = zeros(eltype(eltype(pos)), length(pos), length(orb))
    for (i, R) in enumerate(pos)
        eshells = filter(s -> Int(s.atom.Z) == Zs[i], bs.basis)
        Zh = SpheriCart.compute(sh, R)
        for (col, p) in enumerate(spec)
            sl = filter(s -> s.l == p.l, eshells)
            rad = p.n <= length(sl) ?
                  sum(sl[p.n].coef[k] * exp(-sl[p.n].exp[k] * dot(R, R))
                      for k in eachindex(sl[p.n].exp)) : zero(eltype(P))
            P[i, col] = rad * Zh[p.l^2 + p.l + p.m + 1]
        end
    end
    return P
end

# analytic single-atom overlap of the converted (species-1) orbitals: different (l,m)
# are orthogonal (L2 harmonics), same (l,m) couple radially via the Gaussian moment
# ∫₀^∞ exp(-γr²) r^{2l+2} dr = ½ Γ(l+3/2) γ^{-(l+3/2)}, Γ(l+3/2) = (2l+1)!!/2^{l+1}·√π.
# Grid-free, libcint-independent overlap oracle for single-element sets.
function _analytic_overlap(orb)
    ζ = orb.Rnl.ζ;  D = orb.Rnl.D
    spec = orb.spec;  nb = length(orb)
    S = zeros(Float64, nb, nb)
    dfac(n) = (p = 1.0; for k = 1:2:n; p *= k; end; p)   # n!!
    for i = 1:nb, j = 1:nb
        (spec[i].l == spec[j].l && spec[i].m == spec[j].m) || continue
        l = spec[i].l;  ki = orb.radidx[i];  kj = orb.radidx[j]
        g = 0.5 * dfac(2l + 1) / 2.0^(l + 1) * sqrt(pi)
        acc = 0.0
        for a = 1:size(ζ, 2), b = 1:size(ζ, 2)
            acc += D[ki, a, 1] * D[kj, b, 1] * (ζ[ki, a, 1] + ζ[kj, b, 1])^(-(l + 1.5))
        end
        S[i, j] = g * acc
    end
    return S
end

@testset "single element" begin
    # single atom at the origin; sto-3g/H minimal, 6-31g/C s,p, cc-pvdz/O & def2-svp/O s,p,d
    for (name, el) in (("sto-3g", "H"), ("6-31g", "C"),
                       ("cc-pvdz", "O"), ("def2-svp", "O"))
        @testset "$name / $el" begin
            bs = BasisSet(name, "$el 0.0 0.0 0.0")
            orb = gaussian_orbitals(bs)
            Z = Int(only(bs.atoms).Z)

            @test AOK.nspecies(orb.Rnl) == 1
            @test length(orb) == bs.nbas == sum(2 * s.l + 1 for s in bs.basis)

            # A. values match the GaussianBasis-defined AO (positions → species 1)
            X = [ @SVector randn(3) for _ = 1:12 ]
            ref = _gb_ref(orb, bs, X, fill(Z, length(X)))
            @test all(isfinite, evaluate(orb, X))
            @test evaluate(orb, X) ≈ ref atol = 1e-10
            @test evaluate_ref(orb, X) ≈ ref atol = 1e-10
            @test evaluate_ed(orb, X)[1] ≈ ref atol = 1e-10

            # Float32 conversion + evaluation path
            orb32 = gaussian_orbitals(bs; T = Float32)
            P32 = evaluate(orb32, [ SVector{3, Float32}(x) for x in X ])
            @test eltype(P32) == Float32
            @test P32 ≈ ref atol = 1e-4

            # B. overlap vs GaussianBasis's libcint `overlap`, order-independent
            ev_gb = sort(eigvals(Symmetric(overlap(bs))))
            ev_an = sort(eigvals(Symmetric(_analytic_overlap(orb))))
            @test ev_an ≈ ev_gb rtol = 1e-6
        end
    end
end

@testset "multi element" begin
    # cc-pvdz C, N, O all have the same 3s2p1d structure → one shared spec, no padding
    bs = BasisSet("cc-pvdz", "C 0.0 0.0 0.0\nN 0.0 0.0 1.0\nO 0.0 1.0 0.0")
    orb = gaussian_orbitals(bs)
    @test AOK.nspecies(orb.Rnl) == 3
    @test orb.Rnl.zlist == (ChemicalSpecies(6), ChemicalSpecies(7), ChemicalSpecies(8))
    @test length(orb) == 14                     # 3s + 2p·3 + 1d·5

    species = (ChemicalSpecies(6), ChemicalSpecies(7), ChemicalSpecies(8))
    Xp = [ PState(𝐫 = (@SVector randn(3)), S = rand(species)) for _ = 1:15 ]
    ref = _gb_ref(orb, bs, [x.𝐫 for x in Xp], [atomic_number(x.S) for x in Xp])
    @test evaluate(orb, Xp) ≈ ref atol = 1e-10
    @test evaluate_ref(orb, Xp) ≈ ref atol = 1e-10
    @test evaluate_ed(orb, Xp)[1] ≈ ref atol = 1e-10

    # padding: H (2s1p) + C (3s2p1d) → shared 3s2p1d; H pads the 3rd s, 2nd p, and d
    bs2 = BasisSet("cc-pvdz", "H 0.0 0.0 0.0\nC 0.0 0.0 1.0")
    orb2 = gaussian_orbitals(bs2)
    @test AOK.nspecies(orb2.Rnl) == 2
    @test length(orb2) == 14
    Xh = [ PState(𝐫 = (@SVector randn(3)), S = ChemicalSpecies(1)) for _ = 1:8 ]
    Ph = evaluate(orb2, Xh)
    @test Ph ≈ _gb_ref(orb2, bs2, [x.𝐫 for x in Xh], fill(1, length(Xh))) atol = 1e-10
    # H lacks the 3rd s, 2nd p, and d shells → those orbital columns are exactly 0
    Hpad = [ c for (c, p) in enumerate(orb2.spec)
             if (p.l == 0 && p.n == 3) || (p.l == 1 && p.n == 2) || p.l == 2 ]
    @test all(iszero, Ph[:, Hpad])
    @test !all(iszero, Ph[:, setdiff(1:length(orb2), Hpad)])   # the rest are populated

    # an element may appear at most once
    @test_throws ErrorException gaussian_orbitals(BasisSet("sto-3g", "H 0 0 0\nH 0 0 1"))
end

@testset "convenience constructors" begin
    # (i) gaussian_orbitals(name, elements): same named set for every element.
    # Must match the explicit BasisSet path (positions are ignored by the converter).
    orbA = gaussian_orbitals("cc-pvdz", [:C, :N, :O])
    orbB = gaussian_orbitals(BasisSet("cc-pvdz", "C 0 0 0\nN 0 0 2\nO 0 0 4"))
    @test orbA.Rnl.zlist == orbB.Rnl.zlist
    @test orbA.spec == orbB.spec
    @test Array(orbA.Rnl.ζ) ≈ Array(orbB.Rnl.ζ)
    @test Array(orbA.Rnl.D) ≈ Array(orbB.Rnl.D)

    # element labels accepted as Int / Symbol / String / ChemicalSpecies
    want = (ChemicalSpecies(6), ChemicalSpecies(7), ChemicalSpecies(8))
    for els in ([6, 7, 8], [:C, :N, :O], ["C", "N", "O"], collect(want))
        @test gaussian_orbitals("cc-pvdz", els).Rnl.zlist == want
    end

    # duplicate element / empty list are errors
    @test_throws ErrorException gaussian_orbitals("sto-3g", [:H, :H])
    @test_throws ErrorException gaussian_orbitals("sto-3g", Symbol[])

    # (ii) per-element mix of named sets: cc-pvdz on C, sto-3g on H
    orbM = gaussian_orbitals([:C => "cc-pvdz", "H" => "sto-3g"])
    @test orbM.Rnl.zlist == (ChemicalSpecies(6), ChemicalSpecies(1))
    @test AOK.nspecies(orbM.Rnl) == 2

    X = [ @SVector randn(3) for _ = 1:8 ]
    # C-tagged points reproduce the cc-pvdz C-only orbital exactly (shared spec
    # equals C's own 3s2p1d, so columns line up)
    orbC = gaussian_orbitals(BasisSet("cc-pvdz", "C 0 0 0"))
    @test orbM.spec == orbC.spec
    Xc = [ PState(𝐫 = x, S = ChemicalSpecies(6)) for x in X ]
    @test evaluate(orbM, Xc) ≈ evaluate(orbC, X) atol = 1e-10

    # H-tagged points: only the 1s column is populated (sto-3g H is 1s), and it
    # matches the sto-3g H-only orbital; every padded column is exactly 0
    orbH = gaussian_orbitals(BasisSet("sto-3g", "H 0 0 0"))
    Xh = [ PState(𝐫 = x, S = ChemicalSpecies(1)) for x in X ]
    Ph = evaluate(orbM, Xh)
    c1s = findfirst(p -> p.l == 0 && p.n == 1, orbM.spec)
    @test Ph[:, c1s] ≈ evaluate(orbH, X)[:, 1] atol = 1e-10
    @test all(iszero, Ph[:, setdiff(1:length(orbM), c1s)])

    @test_throws ErrorException gaussian_orbitals(Pair{Symbol, String}[])
end
