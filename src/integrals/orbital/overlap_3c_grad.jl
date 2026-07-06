# Differentiable 3-center Cartesian-Gaussian overlap, mirroring the 2-center
# path in overlap_2c_grad.jl: Lux-style `(ps, st)` calling convention and a
# ChainRules `rrule` for the 3C `batch_overlap` w.r.t. the radial `(ζ, D)`.
# `coef` is recomputed from `ps` per call (`_compiled_from_ps`), so learned
# parameters transfer straight back to the `AtomicOrbitals`.
#
# Gradient decomposition (the overlap is TRIlinear in `coef`):
#   ∂coef, ∂α   from `batch_V_orbital_pb_kernel!` (∂α = the kernel's ∂/∂ζ)
#   ∂D, ∂ζ_norm from `_compile_coef_pb` (shared with 2C)
#   ∂ζ = ∂α + ∂ζ_norm
# Positions are `NoTangent()` (parameters-only), as for 2C.

"""
    batch_overlap(orb::AtomicOrbitals, XA, XB, XC, ps, st) -> Array
    batch_overlap!(out, orb::AtomicOrbitals, XA, XB, XC, ps, st) -> out

Lux-style differentiable 3-center overlap: `coef` is recomputed from the radial
parameters `ps.Rnl = (ζ, D)`, so the result is differentiable w.r.t. `(ζ, D)` via
the `rrule` below. The non-mutating form allocates with the promoted float type
of the positions and `(ζ, D)` (so `ForwardDiff.Dual` parameters flow through).
"""
function batch_overlap(orb::GaussianTypeOrbitals,
                       XA::AbstractVector{<:PState},
                       XB::AbstractVector{<:PState},
                       XC::AbstractVector{<:PState}, ps, st)
    basis = _compiled_from_ps(orb, ps)
    posFT = float(eltype(_position(first(XA))))
    FT = promote_type(posFT, eltype(basis.coef))
    N = basis.nbf_total
    out = zeros(FT, N, N, N, length(XA))
    return batch_overlap!(out, basis, XA, XB, XC)
end

function batch_overlap!(out, orb::GaussianTypeOrbitals,
                        XA::AbstractVector{<:PState},
                        XB::AbstractVector{<:PState},
                        XC::AbstractVector{<:PState}, ps, st;
                        backend = KA.get_backend(out))
    return batch_overlap!(out, _compiled_from_ps(orb, ps), XA, XB, XC;
                          backend = backend)
end

# ---- analytic kernel pullback: ∂coef and ∂α (= ∂/∂ζ through the kernel) ----
#
# Mirrors `batch_V_orbital_kernel!` exactly — same E3 recursion, strides and
# block readback. Trilinearity gives ∂coef as reweights of the primitive-triple
# overlaps `G`; ∂/∂α uses the raised-angular-momentum identity per center,
#   ∂G/∂αₐ = -(G^{a+2x} + G^{a+2y} + G^{a+2z}),
# (and symmetrically for αᵦ, α_c against the SAME combined exponent q), i.e. the
# E3 tables evaluated two orders higher — recursion run to `Lmax+2` per index.
# Species-shared slots → `KA.@atomic` scatter (segmented layout = perf
# follow-up, as for 2C). The raised E3 scratch is (3Lr+3)·(Lr+1)³·3 floats with
# `Lr = Lmax+2` — large per work-item; fine for the small Lmax of orbital bases.
@kernel function batch_V_orbital_pb_kernel!(∂coef, ∂α, @Const(∂out),
                                 @Const(ls), @Const(nbf), @Const(basis_offset),
                                 @Const(coef), @Const(α),
                                 @Const(sidxA), @Const(sidxB), @Const(sidxC),
                                 @Const(posA), @Const(posB), @Const(posC),
                                 ::Val{Lmax}) where {Lmax}
    b, s_a, s_b, s_c = @index(Global, NTuple)

    FT = eltype(∂coef)
    σa = sidxA[b]
    σb = sidxB[b]
    σc = sidxC[b]
    l_a = ls[s_a];     l_b = ls[s_b];     l_c = ls[s_c]
    nbf_a = nbf[s_a];  nbf_b = nbf[s_b]
    row_off  = basis_offset[s_a]
    col_off  = basis_offset[s_b]
    page_off = basis_offset[s_c]
    N1 = nbf_a
    N2 = nbf_b
    K = size(coef, 2)

    # raised-momentum E3-tables (powers up to l+2): same recursion, two orders up
    Lr = Lmax + 2
    E3 = MArray{Tuple{3 * Lr + 3, Lr + 1, Lr + 1, Lr + 1, 3}, FT}(undef)
    la1 = l_a + 3
    lb1 = l_b + 3
    lc1 = l_c + 3

    half_oo = FT(0.5)
    π_FT = FT(π)
    pow15 = FT(1.5)

    A = SVector(posA[1, b], posA[2, b], posA[3, b])
    B = SVector(posB[1, b], posB[2, b], posB[3, b])
    C = SVector(posC[1, b], posC[2, b], posC[3, b])

    # As in 2C, do NOT skip `coef==0` (padded) primitives: a padded slot
    # contributes nothing to the value but has a nonzero ∂out/∂coef, which the
    # true VJP (and ForwardDiff) include. Padding is `ζ=1, D=0` (inert).
    @inbounds for ip in 1:K, jp in 1:K, kp in 1:K
        ca = coef[s_a, ip, σa]
        cb = coef[s_b, jp, σb]
        cc = coef[s_c, kp, σc]
        αa = α[s_a, ip, σa]
        αb = α[s_b, jp, σb]
        αc = α[s_c, kp, σc]

        q, Q, P, μ, ν = _gaussprod_3c(αa, αb, αc, A, B, C)
        oo2q = half_oo / q
        pref0 = (π_FT / q)^pow15

        # same shared recursion as the forward, with raised bounds `l+3`
        _mdE_3c!(E3, la1, lb1, lc1, A, B, C, P, Q, μ, ν, oo2q)

        # contract with the output cotangent, mirroring the forward strides and
        # readback `out[ii,jj,kk] = blk[ii + N1*(jj-1) + N1*N2*(kk-1)]` (the
        # reconstruction recovers `(index1,index2,index3)`). Accumulate into
        # scalars, then one atomic per (ip,jp,kp) and parameter.
        Bsum = zero(FT)   # Σ ∂out · Eprod           (∂coef weight)
        Aa = zero(FT)     # Σ ∂out · ∂G/∂αa / pref0  (∂α for center A)
        Ab = zero(FT)     # Σ ∂out · ∂G/∂αb / pref0  (∂α for center B)
        Ac = zero(FT)     # Σ ∂out · ∂G/∂αc / pref0  (∂α for center C)
        index1 = 1
        for ll1 in l_a:-1:0, n1 in 0:(l_a - ll1)
            m1 = l_a - ll1 - n1
            index2 = 1
            for ll2 in l_b:-1:0, n2 in 0:(l_b - ll2)
                m2 = l_b - ll2 - n2
                index3 = 1
                for ll3 in l_c:-1:0, n3 in 0:(l_c - ll3)
                    m3 = l_c - ll3 - n3
                    lin = index1 + N1 * (index2 - 1) + N1 * N2 * (index3 - 1)
                    ii = (lin - 1) % N1 + 1
                    jj = ((lin - 1) ÷ N1) % N2 + 1
                    kk = (lin - 1) ÷ (N1 * N2) + 1
                    g = ∂out[row_off + ii, col_off + jj, page_off + kk, b]
                    Ex = E3[1, ll1+1, ll2+1, ll3+1, 1]
                    Ey = E3[1, m1+1,  m2+1,  m3+1,  2]
                    Ez = E3[1, n1+1,  n2+1,  n3+1,  3]
                    Eprod = Ex * Ey * Ez
                    dGa = -(E3[1, ll1+3, ll2+1, ll3+1, 1] * Ey * Ez +
                            Ex * E3[1, m1+3, m2+1, m3+1, 2] * Ez +
                            Ex * Ey * E3[1, n1+3, n2+1, n3+1, 3])
                    dGb = -(E3[1, ll1+1, ll2+3, ll3+1, 1] * Ey * Ez +
                            Ex * E3[1, m1+1, m2+3, m3+1, 2] * Ez +
                            Ex * Ey * E3[1, n1+1, n2+3, n3+1, 3])
                    dGc = -(E3[1, ll1+1, ll2+1, ll3+3, 1] * Ey * Ez +
                            Ex * E3[1, m1+1, m2+1, m3+3, 2] * Ez +
                            Ex * Ey * E3[1, n1+1, n2+1, n3+3, 3])
                    Bsum += g * Eprod
                    Aa += g * dGa
                    Ab += g * dGb
                    Ac += g * dGc
                    index3 += 1
                end
                index2 += 1
            end
            index1 += 1
        end
        KA.@atomic ∂coef[s_a, ip, σa] += cb * cc * pref0 * Bsum
        KA.@atomic ∂coef[s_b, jp, σb] += ca * cc * pref0 * Bsum
        KA.@atomic ∂coef[s_c, kp, σc] += ca * cb * pref0 * Bsum
        KA.@atomic ∂α[s_a, ip, σa]   += ca * cb * cc * pref0 * Aa
        KA.@atomic ∂α[s_b, jp, σb]   += ca * cb * cc * pref0 * Ab
        KA.@atomic ∂α[s_c, kp, σc]   += ca * cb * cc * pref0 * Ac
    end
end

# run the pullback kernel; returns (∂coef, ∂α) shaped like `basis.coef`.
# Positions are scaled to native Bohr exactly as in the forward.
function _overlap_3c_pb_coef_alpha(∂V, basis::CartesianGTOBasis,
                                   XA::AbstractVector{<:PState},
                                   XB::AbstractVector{<:PState},
                                   XC::AbstractVector{<:PState};
                                   backend = KA.get_backend(∂V))
    FT = float(promote_type(eltype(∂V), eltype(basis.coef)))
    B = length(XA)
    ArrayCtor = typeof(∂V).name.wrapper
    s = FT(basis.lengthscale)
    posA = _cgto_posmat(XA, FT);  s == one(s) || (posA .*= s)
    posB = _cgto_posmat(XB, FT);  s == one(s) || (posB .*= s)
    posC = _cgto_posmat(XC, FT);  s == one(s) || (posC .*= s)
    posA_d = ArrayCtor(posA)
    posB_d = ArrayCtor(posB)
    posC_d = ArrayCtor(posC)
    coef_d = ArrayCtor(FT.(basis.coef))
    α_d    = ArrayCtor(FT.(basis.ζ))
    sA_d   = ArrayCtor(_cgto_sidx(basis, XA))
    sB_d   = ArrayCtor(_cgto_sidx(basis, XB))
    sC_d   = ArrayCtor(_cgto_sidx(basis, XC))
    ∂V_d   = ArrayCtor(FT.(∂V))
    ∂coef  = fill!(similar(coef_d), zero(FT))
    ∂α     = fill!(similar(α_d), zero(FT))
    batch_V_orbital_pb_kernel!(backend)(∂coef, ∂α, ∂V_d, basis.ls, basis.nbf,
            basis.basis_offset, coef_d, α_d, sA_d, sB_d, sC_d,
            posA_d, posB_d, posC_d, Val(basis.Lmax);
            ndrange = (B, basis.nshells, basis.nshells, basis.nshells))
    KA.synchronize(backend)
    return ∂coef, ∂α
end

# ---- parameter pullback + ChainRules rrule ----

"""
    batch_overlap_pb_ps(∂V, orb::AtomicOrbitals, XA, XB, XC, ps, st)
        -> (Rnl=(ζ=∂ζ, D=∂D), Ylm=())

Parameter pullback of the 3-center overlap w.r.t. the radial `(ζ, D)`: combines
the analytic kernel pullback (`∂coef`, `∂α`) with the normalization pullback.
"""
function batch_overlap_pb_ps(∂V, orb::GaussianTypeOrbitals,
                             XA::AbstractVector{<:PState},
                             XB::AbstractVector{<:PState},
                             XC::AbstractVector{<:PState}, ps, st)
    basis = _compiled_from_ps(orb, ps)
    ∂coef, ∂α = _overlap_3c_pb_coef_alpha(∂V, basis, XA, XB, XC)
    ∂D, ∂ζ_norm = _compile_coef_pb(∂coef, ps.Rnl.ζ, ps.Rnl.D, basis.ls)
    ∂ζ = ∂α .+ ∂ζ_norm
    return (Rnl = (ζ = ∂ζ, D = ∂D), Ylm = NamedTuple())
end

# Differentiable w.r.t. params only; positions are `NoTangent()` (as for 2C).
function rrule(::typeof(batch_overlap), orb::GaussianTypeOrbitals,
               XA::AbstractVector{<:PState}, XB::AbstractVector{<:PState},
               XC::AbstractVector{<:PState}, ps, st)
    V = batch_overlap(orb, XA, XB, XC, ps, st)
    function _pb(_∂)
        ∂V = unthunk(_∂)
        ∂ps = batch_overlap_pb_ps(∂V, orb, XA, XB, XC, ps, st)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                ∂ps, NoTangent())
    end
    return V, _pb
end
