# Independent Gauss–Hermite overlap oracle for the Cartesian-Gaussian overlap
# kernels. It integrates the Cartesian AO products numerically and writes each
# shell block with the `nbf = (l+1)(l+2)/2` stride, sharing none of the kernels'
# machinery — no McMurchie–Davidson E-recursion and no flat block indexing — so it
# is a genuinely independent ground truth (the scalar `Reference`, by contrast,
# reuses the kernels' flat block layout). It was built to catch an l ≥ 2 write-
# stride aliasing bug (d/f blocks written with a `2l+1` stride instead of `nbf`).
#
# Gauss–Hermite quadrature is exact for the polynomial×Gaussian integrands here
# (∫ p(x) e^{-x²} dx is exact for deg p ≤ 2n−1), so the oracle is machine-precise.

using FastGaussQuadrature: gausshermite
using StaticArrays
import AtomicOrbitalKernels.Reference: ang2bohr

# GH nodes/weights (weight e^{-t²}). Exact for a 1D integrand of polynomial
# degree ≤ 2n−1; the highest total degree is 3·Lmax (the 3-center triple), so
# n = 3·Lmax + 2 is comfortably exact for both the 2C and 3C oracles.
_gh_nodes(Lmax) = gausshermite(3 * Lmax + 2)

# 1D two-center overlap ∫ (x−Ax)^la (x−Bx)^lb e^{−a(x−Ax)²} e^{−b(x−Bx)²} dx via
# the Gaussian product theorem (p, Px, μ) + the substitution t = √p (x − Px).
function _quad_S1d(la, lb, a, b, Ax, Bx, t, w)
    p  = a + b
    Px = (a * Ax + b * Bx) / p
    μ  = a * b / p
    sp = sqrt(p)
    pref = exp(-μ * (Ax - Bx)^2) / sp
    s = zero(pref)
    @inbounds for k in eachindex(t)
        x = Px + t[k] / sp
        s += w[k] * (x - Ax)^la * (x - Bx)^lb
    end
    return pref * s
end

# 1D three-center overlap: iterated Gaussian product (a,b)→(p,Px) then (p,c)→
# (q,Qx), substitution t = √q (x − Qx). Exact under the same GH rule.
function _quad_S1d_3(la, lb, lc, a, b, c, Ax, Bx, Cx, t, w)
    p   = a + b
    Px  = (a * Ax + b * Bx) / p
    μab = a * b / p
    q   = p + c
    Qx  = (p * Px + c * Cx) / q
    ν   = p * c / q
    sq  = sqrt(q)
    pref = exp(-μab * (Ax - Bx)^2 - ν * (Px - Cx)^2) / sq
    s = zero(pref)
    @inbounds for k in eachindex(t)
        x = Qx + t[k] / sq
        s += w[k] * (x - Ax)^la * (x - Bx)^lb * (x - Cx)^lc
    end
    return pref * s
end

# per-shell Cartesian function count and cumulative offsets (Cartesian layout)
function _cart_offsets(BS)
    nshells = length(BS.basis)
    nbf = Int[(BS.basis[i].l + 1) * (BS.basis[i].l + 2) ÷ 2 for i in 1:nshells]
    offset = zeros(Int, nshells + 1)
    for i in 1:nshells
        offset[i+1] = offset[i] + nbf[i]
    end
    return nbf, offset
end

# fill the (nbf_a × nbf_b) block for a shell pair, contracting over primitive
# pairs. Component-loop order matches `generate_S_pair!`, but the block is a plain
# 2D view so the write uses the correct `nbf_a` stride.
function _quad_pair_block!(block, sa, sb, A, B, t, w)
    am1 = sa.l
    am2 = sb.l
    fill!(block, zero(eltype(block)))
    for (ca, a) in zip(sa.coef, sa.exp), (cb, b) in zip(sb.coef, sb.exp)
        index1 = 1
        for l1 in am1:-1:0, n1 in 0:am1-l1
            m1 = am1 - l1 - n1
            index2 = 1
            for l2 in am2:-1:0, n2 in 0:am2-l2
                m2 = am2 - l2 - n2
                Sx = _quad_S1d(l1, l2, a, b, A[1], B[1], t, w)
                Sy = _quad_S1d(m1, m2, a, b, A[2], B[2], t, w)
                Sz = _quad_S1d(n1, n2, a, b, A[3], B[3], t, w)
                block[index1, index2] += ca * cb * Sx * Sy * Sz
                index2 += 1
            end
            index1 += 1
        end
    end
    return block
end

"""
    quad_batch_S_pair!(out, BS, posA, posB)

Independent Gauss–Hermite ground truth for the batched 2-center overlap. Same
signature/units as `Reference.batch_S_pair_ref!` (plain Å position matrices,
converted to Bohr internally; `coef` taken verbatim from the shells), but writes
each block with the correct `nbf` stride.
"""
function quad_batch_S_pair!(out::AbstractArray{T,3}, BS,
                            posA::AbstractMatrix, posB::AbstractMatrix) where {T}
    nbf, offset = _cart_offsets(BS)
    nshells = length(BS.basis)
    Ntot = offset[end]
    @assert size(out, 1) == Ntot && size(out, 2) == Ntot
    Lmax = maximum(s.l for s in BS.basis)
    t, w = _gh_nodes(Lmax)

    fill!(out, zero(T))
    for b in 1:size(posA, 2)
        𝐫A = SVector{3,Float64}(posA[1, b], posA[2, b], posA[3, b]) .* ang2bohr
        𝐫B = SVector{3,Float64}(posB[1, b], posB[2, b], posB[3, b]) .* ang2bohr
        for s_a in 1:nshells, s_b in 1:nshells
            block = view(out, offset[s_a]+1:offset[s_a]+nbf[s_a],
                              offset[s_b]+1:offset[s_b]+nbf[s_b], b)
            _quad_pair_block!(block, BS.basis[s_a], BS.basis[s_b], 𝐫A, 𝐫B, t, w)
        end
    end
    return out
end

# fill the (nbf_a × nbf_b × nbf_c) block for a shell triple. As in the 2C case,
# the block is a plain 3D view, so writes use the correct Cartesian strides.
function _quad_triple_block!(block, sa, sb, sc, A, B, C, t, w)
    am1 = sa.l
    am2 = sb.l
    am3 = sc.l
    fill!(block, zero(eltype(block)))
    for (ca, a) in zip(sa.coef, sa.exp), (cb, b) in zip(sb.coef, sb.exp),
        (cc, c) in zip(sc.coef, sc.exp)
        index1 = 1
        for l1 in am1:-1:0, n1 in 0:am1-l1
            m1 = am1 - l1 - n1
            index2 = 1
            for l2 in am2:-1:0, n2 in 0:am2-l2
                m2 = am2 - l2 - n2
                index3 = 1
                for l3 in am3:-1:0, n3 in 0:am3-l3
                    m3 = am3 - l3 - n3
                    Sx = _quad_S1d_3(l1, l2, l3, a, b, c, A[1], B[1], C[1], t, w)
                    Sy = _quad_S1d_3(m1, m2, m3, a, b, c, A[2], B[2], C[2], t, w)
                    Sz = _quad_S1d_3(n1, n2, n3, a, b, c, A[3], B[3], C[3], t, w)
                    block[index1, index2, index3] += ca * cb * cc * Sx * Sy * Sz
                    index3 += 1
                end
                index2 += 1
            end
            index1 += 1
        end
    end
    return block
end

"""
    quad_batch_V_triple!(out, BS, posA, posB, posC)

Independent Gauss–Hermite ground truth for the batched 3-center overlap. Same
signature/units as `Reference.batch_V_triple_ref!`, with correct `nbf` strides.
"""
function quad_batch_V_triple!(out::AbstractArray{T,4}, BS,
                              posA::AbstractMatrix, posB::AbstractMatrix,
                              posC::AbstractMatrix) where {T}
    nbf, offset = _cart_offsets(BS)
    nshells = length(BS.basis)
    Ntot = offset[end]
    @assert size(out, 1) == Ntot && size(out, 2) == Ntot && size(out, 3) == Ntot
    Lmax = maximum(s.l for s in BS.basis)
    t, w = _gh_nodes(Lmax)

    fill!(out, zero(T))
    for b in 1:size(posA, 2)
        𝐫A = SVector{3,Float64}(posA[1, b], posA[2, b], posA[3, b]) .* ang2bohr
        𝐫B = SVector{3,Float64}(posB[1, b], posB[2, b], posB[3, b]) .* ang2bohr
        𝐫C = SVector{3,Float64}(posC[1, b], posC[2, b], posC[3, b]) .* ang2bohr
        for s_a in 1:nshells, s_b in 1:nshells, s_c in 1:nshells
            block = view(out, offset[s_a]+1:offset[s_a]+nbf[s_a],
                              offset[s_b]+1:offset[s_b]+nbf[s_b],
                              offset[s_c]+1:offset[s_c]+nbf[s_c], b)
            _quad_triple_block!(block, BS.basis[s_a], BS.basis[s_b],
                                BS.basis[s_c], 𝐫A, 𝐫B, 𝐫C, t, w)
        end
    end
    return out
end
