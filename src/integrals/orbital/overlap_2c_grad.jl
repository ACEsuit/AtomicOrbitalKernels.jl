# Differentiable 2-center Cartesian-Gaussian overlap (Stage 4).
#
# Adds the Lux-style `(ps, st)` calling convention and a ChainRules `rrule` for
# `batch_overlap` on a Gaussian `AtomicOrbitals` basis, mirroring the orbital
# `evaluate` path. The learnable parameters are the orbital radials' `(ζ, D)`
# (`ps.Rnl = (ζ, D)`, shared verbatim with `evaluate`); `coef` is recomputed from
# `ps` on every call (`_compile_coef`), so the parameters stay the single source
# of truth and learned values transfer straight back to the `AtomicOrbitals`.
#
# Gradient decomposition (the overlap is bilinear in `coef`, and `coef` is a
# differentiable function of `(ζ, D)`):
#   ∂coef, ∂α   from `batch_S_orbital_pb_kernel!` (∂α = the kernel's ∂/∂ζ)
#   ∂D, ∂ζ_norm from `_compile_coef_pb` (the spherical→Cartesian normalization)
#   ∂ζ = ∂α + ∂ζ_norm   (ζ enters both the kernel and the normalization;
#                        D enters only through `coef`)
# Positions are treated as `NoTangent()` for now (parameters-only); an atom-center
# gradient would add a position-derivative kernel + `VState` cotangents here.

# Build a `CartesianGTOBasis` whose `ζ`/`coef` come from `ps` (not from the
# basis's stored arrays), preserving the param element type (e.g. `Dual`) so the
# forward stays differentiable. Structure is param-free and cheap to recompute.
function _compiled_from_ps(orb::GaussianTypeOrbitals, ps)
    str  = _compile_struct(orb)
    ζ    = ps.Rnl.ζ
    coef = _compile_coef(ζ, ps.Rnl.D, str.ls)
    T    = promote_type(eltype(ζ), eltype(coef))
    return CartesianGTOBasis{T, Vector{Int}, Array{T,3}, str.NZ, eltype(str.zlist)}(
        str.Lmax, str.nshells, str.K, str.ls, str.nbf, str.basis_offset,
        str.nbf_total, T.(ζ), T.(coef), str.zlist, T(orb.lengthscale))
end

"""
    batch_overlap(orb::AtomicOrbitals, XA, XB, ps, st) -> Array
    batch_overlap!(out, orb::AtomicOrbitals, XA, XB, ps, st) -> out

Lux-style differentiable 2-center overlap: `coef` is recomputed from the radial
parameters `ps.Rnl = (ζ, D)`, so the result is differentiable w.r.t. `(ζ, D)` via
the `rrule` below. The non-mutating form allocates with the promoted float type
of the positions and `(ζ, D)` (so `ForwardDiff.Dual` parameters flow through).
"""
function batch_overlap(orb::GaussianTypeOrbitals,
                       XA::AbstractVector{<:PState},
                       XB::AbstractVector{<:PState}, ps, st)
    basis = _compiled_from_ps(orb, ps)
    posFT = float(eltype(_position(first(XA))))
    FT = promote_type(posFT, eltype(basis.coef))
    N = basis.nbf_total
    out = zeros(FT, N, N, length(XA))
    return batch_overlap!(out, basis, XA, XB)
end

function batch_overlap!(out, orb::GaussianTypeOrbitals,
                        XA::AbstractVector{<:PState},
                        XB::AbstractVector{<:PState}, ps, st;
                        backend = KA.get_backend(out))
    return batch_overlap!(out, _compiled_from_ps(orb, ps), XA, XB; backend = backend)
end

# ---- analytic kernel pullback: ∂coef and ∂α (= ∂/∂ζ through the kernel) ----
#
# Mirrors `batch_S_orbital_kernel!` exactly — same MD recursion, same `N1=nbf_a`
# stride and block readback — so the gradient matches the forward bit for bit.
# The overlap is bilinear in `coef` (∂coef is a reweight of
# the same primitive-pair overlaps `G`), and ∂/∂α uses the raised-angular-momentum
# identity ∂/∂αₐ e^{-αₐr²} = -r²e^{-αₐr²}, i.e.
#   ∂G/∂αₐ = -(G^{a+2x} + G^{a+2y} + G^{a+2z}),
# overlaps with the bra monomial raised by 2 on each axis (and symmetrically for
# αᵦ). Those are the same E-coefficients evaluated two orders higher, so the
# recursion is run to `Lmax+2`. Several work-items hit the same species-shared
# slot → `KA.@atomic` scatter (a segmented layout is a perf follow-up).
@kernel function batch_S_orbital_pb_kernel!(∂coef, ∂α, @Const(∂out),
                                 @Const(ls), @Const(nbf), @Const(basis_offset),
                                 @Const(coef), @Const(α),
                                 @Const(sidxA), @Const(sidxB),
                                 @Const(posA), @Const(posB),
                                 ::Val{Lmax}) where {Lmax}
    b, s_a, s_b = @index(Global, NTuple)

    FT = eltype(∂coef)
    σa = sidxA[b]
    σb = sidxB[b]
    l_a = ls[s_a]
    l_b = ls[s_b]
    nbf_a = nbf[s_a]
    row_off = basis_offset[s_a]
    col_off = basis_offset[s_b]
    N1 = nbf_a
    K = size(coef, 2)

    # raised-momentum E-tables (powers up to l+2): same recursion, two orders up
    Lr = Lmax + 2
    E = MArray{Tuple{2 * Lr + 2, Lr + 1, Lr + 1, 3}, FT}(undef)
    Tmax = 2 * Lr + 2
    la1 = l_a + 3
    lb1 = l_b + 3

    half_oo = FT(0.5)
    π_FT = FT(π)
    pow15 = FT(1.5)

    Ax = posA[1, b]; Ay = posA[2, b]; Az = posA[3, b]
    Bx = posB[1, b]; By = posB[2, b]; Bz = posB[3, b]

    # Unlike the forward kernel, we do NOT skip `coef==0` (padded) primitives:
    # a padded slot contributes nothing to the value (`ca·cb = 0`) but has a
    # nonzero derivative `∂out/∂coef = cb·G`, which the true VJP (and ForwardDiff)
    # include. Padding is `ζ=1, D=0` (inert, positive ζ) so the geometry is
    # well-defined; gradients on padded slots are masked by the caller, matching
    # the radial-pullback convention.
    @inbounds for ip in 1:K, jp in 1:K
        ca = coef[s_a, ip, σa]
        cb = coef[s_b, jp, σb]
        αa = α[s_a, ip, σa]
        αb = α[s_b, jp, σb]

        p = αa + αb
        Px = (αa * Ax + αb * Bx) / p
        Py = (αa * Ay + αb * By) / p
        Pz = (αa * Az + αb * Bz) / p
        μ = αa * αb / p
        oo2p = half_oo / p
        pref0 = (π_FT / p)^pow15

        for ax in 1:3
            P_ax = ax == 1 ? Px : (ax == 2 ? Py : Pz)
            A_ax = ax == 1 ? Ax : (ax == 2 ? Ay : Az)
            B_ax = ax == 1 ? Bx : (ax == 2 ? By : Bz)
            AB = A_ax - B_ax
            PA = P_ax - A_ax
            PB = P_ax - B_ax

            for jb in 1:lb1, ia in 1:la1
                ab = ia + jb
                if ab <= Tmax
                    E[ab, ia, jb, ax] = zero(FT)
                end
                if ab + 1 <= Tmax
                    E[ab+1, ia, jb, ax] = zero(FT)
                end
            end
            E[1, 1, 1, ax] = exp(-μ * AB * AB)

            for i in 2:la1
                E[1, i, 1, ax] = PA * E[1, i-1, 1, ax] + E[2, i-1, 1, ax]
                for t in 2:i
                    tFT = FT(t)
                    E[t, i, 1, ax] = PA * E[t, i-1, 1, ax] + tFT * E[t+1, i-1, 1, ax] +
                                     oo2p * E[t-1, i-1, 1, ax]
                end
            end
            for j in 2:lb1
                E[1, 1, j, ax] = PB * E[1, 1, j-1, ax] + E[2, 1, j-1, ax]
                for t in 2:j
                    tFT = FT(t)
                    E[t, 1, j, ax] = PB * E[t, 1, j-1, ax] + tFT * E[t+1, 1, j-1, ax] +
                                     oo2p * E[t-1, 1, j-1, ax]
                end
                for i in 2:la1
                    E[1, i, j, ax] = PA * E[1, i-1, j, ax] + E[2, i-1, j, ax]
                    for t in 2:i+j-1
                        tFT = FT(t)
                        E[t, i, j, ax] = PA * E[t, i-1, j, ax] + tFT * E[t+1, i-1, j, ax] +
                                         oo2p * E[t-1, i-1, j, ax]
                    end
                end
            end
        end

        # contract with the output cotangent, mirroring the forward `N1` stride
        # and block readback `out[i,j] = blk[i + nbf_a*(j-1)]` (with N1 = nbf_a the
        # reconstruction recovers `(i,j) = (index1,index2)`). Accumulate the block
        # into scalars, then one atomic per (ip,jp) and parameter.
        Bsum = zero(FT)   # Σ ∂out · Eprod          (∂coef weight)
        Aa = zero(FT)     # Σ ∂out · ∂G/∂αa / pref0 (∂α for bra)
        Ab = zero(FT)     # Σ ∂out · ∂G/∂αb / pref0 (∂α for ket)
        index1 = 1
        for ll1 in l_a:-1:0
            for n1 in 0:(l_a - ll1)
                m1 = l_a - ll1 - n1
                index2 = 1
                for ll2 in l_b:-1:0
                    for n2 in 0:(l_b - ll2)
                        m2 = l_b - ll2 - n2
                        lin = index1 + N1 * (index2 - 1)
                        i = (lin - 1) % nbf_a + 1
                        j = (lin - 1) ÷ nbf_a + 1
                        g = ∂out[row_off + i, col_off + j, b]
                        Ex = E[1, ll1+1, ll2+1, 1]
                        Ey = E[1, m1+1, m2+1, 2]
                        Ez = E[1, n1+1, n2+1, 3]
                        Eprod = Ex * Ey * Ez
                        Exa = E[1, ll1+3, ll2+1, 1]
                        Eya = E[1, m1+3, m2+1, 2]
                        Eza = E[1, n1+3, n2+1, 3]
                        Exb = E[1, ll1+1, ll2+3, 1]
                        Eyb = E[1, m1+1, m2+3, 2]
                        Ezb = E[1, n1+1, n2+3, 3]
                        dGa = -(Exa * Ey * Ez + Ex * Eya * Ez + Ex * Ey * Eza)
                        dGb = -(Exb * Ey * Ez + Ex * Eyb * Ez + Ex * Ey * Ezb)
                        Bsum += g * Eprod
                        Aa += g * dGa
                        Ab += g * dGb
                        index2 += 1
                    end
                end
                index1 += 1
            end
        end
        KA.@atomic ∂coef[s_a, ip, σa] += cb * pref0 * Bsum
        KA.@atomic ∂coef[s_b, jp, σb] += ca * pref0 * Bsum
        KA.@atomic ∂α[s_a, ip, σa]   += ca * cb * pref0 * Aa
        KA.@atomic ∂α[s_b, jp, σb]   += ca * cb * pref0 * Ab
    end
end

# run the pullback kernel; returns (∂coef, ∂α) shaped like `basis.coef`. Positions
# are scaled to native Bohr exactly as in the forward `batch_overlap!`.
function _overlap_pb_coef_alpha(∂S, basis::CartesianGTOBasis,
                                XA::AbstractVector{<:PState},
                                XB::AbstractVector{<:PState};
                                backend = KA.get_backend(∂S))
    FT = float(promote_type(eltype(∂S), eltype(basis.coef)))
    B = length(XA)
    ArrayCtor = typeof(∂S).name.wrapper
    s = FT(basis.lengthscale)
    posA = _cgto_posmat(XA, FT);  s == one(s) || (posA .*= s)
    posB = _cgto_posmat(XB, FT);  s == one(s) || (posB .*= s)
    posA_d = ArrayCtor(posA)
    posB_d = ArrayCtor(posB)
    coef_d = ArrayCtor(FT.(basis.coef))
    α_d    = ArrayCtor(FT.(basis.ζ))
    sA_d   = ArrayCtor(_cgto_sidx(basis, XA))
    sB_d   = ArrayCtor(_cgto_sidx(basis, XB))
    ∂S_d   = ArrayCtor(FT.(∂S))
    ∂coef  = fill!(similar(coef_d), zero(FT))
    ∂α     = fill!(similar(α_d), zero(FT))
    batch_S_orbital_pb_kernel!(backend)(∂coef, ∂α, ∂S_d, basis.ls, basis.nbf,
            basis.basis_offset, coef_d, α_d, sA_d, sB_d, posA_d, posB_d,
            Val(basis.Lmax); ndrange = (B, basis.nshells, basis.nshells))
    KA.synchronize(backend)
    return ∂coef, ∂α
end

# ---- normalization pullback: ∂coef → (∂D, ∂ζ_norm) ----
#
# Closed-form VJP of `coef_k = D_k/√normsq`, `normsq = c·Σ_ij D_iD_j/(ζ_i+ζ_j)^e`,
# `c = π^{3/2}(2l-1)!!/2^l`, `e = l+3/2`, per shell/species over the small `K`:
#   ∂D_m = ∂coef_m/n − g_m·dotcD/(2n³),  g_m = 2c·Σ_j D_j/(ζ_m+ζ_j)^e
#   ∂ζ_m =           − h_m·dotcD/(2n³),  h_m = −2c·e·D_m·Σ_j D_j/(ζ_m+ζ_j)^{e+1}
# with dotcD = Σ_i ∂coef_i·D_i, n = √normsq.
function _compile_coef_pb(∂coef, ζ, D, ls)
    nshells, K, NZ = size(ζ)
    T = float(promote_type(eltype(∂coef), eltype(ζ), eltype(D)))
    ∂D = zeros(T, nshells, K, NZ)
    ∂ζ = zeros(T, nshells, K, NZ)
    for σ in 1:NZ, k in 1:nshells
        l = ls[k]
        Dk = @view D[k, :, σ]
        ζk = @view ζ[k, :, σ]
        ∂ck = @view ∂coef[k, :, σ]
        ns = _cart_normsq(Dk, ζk, l)
        ns > 0 || continue
        n = sqrt(T(ns))
        c = (T(π)^T(3//2) * (l == 0 ? one(T) : T(_cart_dfac(2l - 1)))) / T(2)^l
        e = T(l) + T(3//2)
        dotcD = zero(T)
        for i in 1:K
            dotcD += T(∂ck[i]) * T(Dk[i])
        end
        s2n3 = dotcD / (2 * n^3)
        for m in 1:K
            gm = zero(T)
            hm = zero(T)
            for j in 1:K
                den = T(ζk[m]) + T(ζk[j])
                gm += T(Dk[j]) / den^e
                hm += T(Dk[j]) / den^(e + 1)
            end
            gm *= 2 * c
            hm *= -2 * c * e * T(Dk[m])
            ∂D[k, m, σ] = T(∂ck[m]) / n - gm * s2n3
            ∂ζ[k, m, σ] = -hm * s2n3
        end
    end
    return ∂D, ∂ζ
end

# ---- parameter pullback + ChainRules rrule ----

"""
    pullback_ps(∂S, orb::AtomicOrbitals, XA, XB, ps, st) -> (Rnl=(ζ=∂ζ, D=∂D), Ylm=())

Parameter pullback of the 2-center overlap w.r.t. the radial `(ζ, D)`: combines
the analytic kernel pullback (`∂coef`, `∂α`) with the normalization pullback.
"""
function pullback_ps(∂S, orb::GaussianTypeOrbitals,
                     XA::AbstractVector{<:PState},
                     XB::AbstractVector{<:PState}, ps, st)
    basis = _compiled_from_ps(orb, ps)
    ∂coef, ∂α = _overlap_pb_coef_alpha(∂S, basis, XA, XB)
    ∂D, ∂ζ_norm = _compile_coef_pb(∂coef, ps.Rnl.ζ, ps.Rnl.D, basis.ls)
    ∂ζ = ∂α .+ ∂ζ_norm
    return (Rnl = (ζ = ∂ζ, D = ∂D), Ylm = NamedTuple())
end

# Differentiable w.r.t. params only; positions are `NoTangent()` (the hook for a
# future atom-center gradient is a position-derivative kernel + `VState`
# cotangents, paralleling the orbital `evaluate` rrule).
function rrule(::typeof(batch_overlap), orb::GaussianTypeOrbitals,
               XA::AbstractVector{<:PState}, XB::AbstractVector{<:PState}, ps, st)
    S = batch_overlap(orb, XA, XB, ps, st)
    function _pb(_∂)
        ∂S = unthunk(_∂)
        ∂ps = pullback_ps(∂S, orb, XA, XB, ps, st)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), ∂ps, NoTangent())
    end
    return S, _pb
end
