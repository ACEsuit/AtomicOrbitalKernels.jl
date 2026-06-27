# Batched 2-center Cartesian-Gaussian overlap for a species-aware
# `CompiledOrbitalBasis`. Mirrors `batch_S_kernel!` (same McMurchie–Davidson
# E-recursion + contraction + block write, including the `N1 = 2l_a+1` linear
# stride) but reads per-species primitives: the bra shells use species `sidxA[b]`
# and the ket shells use species `sidxB[b]`, so each batch element is an overlap
# between an atom of one species and an atom of another.
#
# Primitive slots are uniform (`K`, padded with `coef=0`); zero-coef pairs are
# skipped. The shared `(n,l)` spec makes every species' shell structure identical,
# so the output block is `nbf_total × nbf_total` for any species pair.
#
# The shared recursion will be factored out of `batch_S_kernel!` / this kernel in
# the DRY-cleanup stage; kept separate for now while the layouts settle.

@kernel function batch_S_orbital_kernel!(out::AbstractArray{FT,3},
                                 @Const(ls), @Const(nbf), @Const(basis_offset),
                                 @Const(coef), @Const(α),
                                 @Const(sidxA), @Const(sidxB),
                                 @Const(posA), @Const(posB),
                                 ::Val{Lmax}) where {FT,Lmax}
    b, s_a, s_b = @index(Global, NTuple)

    σa = sidxA[b]
    σb = sidxB[b]
    l_a = ls[s_a]
    l_b = ls[s_b]
    nbf_a = nbf[s_a]
    nbf_b = nbf[s_b]
    row_off = basis_offset[s_a]
    col_off = basis_offset[s_b]
    N1 = 2 * l_a + 1
    K = size(coef, 2)

    E   = MArray{Tuple{2 * Lmax + 2, Lmax + 1, Lmax + 1, 3}, FT}(undef)
    blk = MArray{Tuple{((Lmax + 1) * (Lmax + 2) ÷ 2)^2}, FT}(undef)
    Tmax = 2 * Lmax + 2

    half_oo = FT(0.5)
    π_FT = FT(π)
    pow15 = FT(1.5)

    @inbounds for k in 1:nbf_a*nbf_b
        blk[k] = zero(FT)
    end

    Ax = posA[1, b]; Ay = posA[2, b]; Az = posA[3, b]
    Bx = posB[1, b]; By = posB[2, b]; Bz = posB[3, b]

    @inbounds for ip in 1:K, jp in 1:K
        ca = coef[s_a, ip, σa]
        cb = coef[s_b, jp, σb]
        (ca == zero(FT) || cb == zero(FT)) && continue
        αa = α[s_a, ip, σa]
        αb = α[s_b, jp, σb]

        p = αa + αb
        Px = (αa * Ax + αb * Bx) / p
        Py = (αa * Ay + αb * By) / p
        Pz = (αa * Az + αb * Bz) / p
        μ = αa * αb / p
        oo2p = half_oo / p
        prefac = ca * cb * (π_FT / p)^pow15

        for ax in 1:3
            P_ax = ax == 1 ? Px : (ax == 2 ? Py : Pz)
            A_ax = ax == 1 ? Ax : (ax == 2 ? Ay : Az)
            B_ax = ax == 1 ? Bx : (ax == 2 ? By : Bz)
            AB = A_ax - B_ax
            PA = P_ax - A_ax
            PB = P_ax - B_ax

            for jb in 1:(l_b + 1), ia in 1:(l_a + 1)
                ab = ia + jb
                if ab <= Tmax
                    E[ab, ia, jb, ax] = zero(FT)
                end
                if ab + 1 <= Tmax
                    E[ab+1, ia, jb, ax] = zero(FT)
                end
            end
            E[1, 1, 1, ax] = exp(-μ * AB * AB)

            for i in 2:(l_a + 1)
                E[1, i, 1, ax] = PA * E[1, i-1, 1, ax] + E[2, i-1, 1, ax]
                for t in 2:i
                    tFT = FT(t)
                    E[t, i, 1, ax] = PA * E[t, i-1, 1, ax] + tFT * E[t+1, i-1, 1, ax] + oo2p * E[t-1, i-1, 1, ax]
                end
            end
            for j in 2:(l_b + 1)
                E[1, 1, j, ax] = PB * E[1, 1, j-1, ax] + E[2, 1, j-1, ax]
                for t in 2:j
                    tFT = FT(t)
                    E[t, 1, j, ax] = PB * E[t, 1, j-1, ax] + tFT * E[t+1, 1, j-1, ax] + oo2p * E[t-1, 1, j-1, ax]
                end
                for i in 2:(l_a + 1)
                    E[1, i, j, ax] = PA * E[1, i-1, j, ax] + E[2, i-1, j, ax]
                    for t in 2:i+j-1
                        tFT = FT(t)
                        E[t, i, j, ax] = PA * E[t, i-1, j, ax] + tFT * E[t+1, i-1, j, ax] + oo2p * E[t-1, i-1, j, ax]
                    end
                end
            end
        end

        # Contraction: flat (nbf_a × nbf_b) block at `index1 + N1*(index2-1)`,
        # N1 = 2*l_a + 1. Matches `batch_S_kernel!` (and Reference) bit-for-bit.
        index1 = 1
        for ll1 in l_a:-1:0
            for n1 in 0:(l_a - ll1)
                m1 = l_a - ll1 - n1
                index2 = 1
                for ll2 in l_b:-1:0
                    for n2 in 0:(l_b - ll2)
                        m2 = l_b - ll2 - n2
                        lin = index1 + N1 * (index2 - 1)
                        blk[lin] += E[1, ll1+1, ll2+1, 1] *
                                    E[1, m1+1, m2+1, 2] *
                                    E[1, n1+1, n2+1, 3] * prefac
                        index2 += 1
                    end
                end
                index1 += 1
            end
        end
    end

    @inbounds for j in 1:nbf_b, i in 1:nbf_a
        out[row_off + i, col_off + j, b] = blk[i + nbf_a * (j - 1)]
    end
end

# Input is a vector of DecoratedParticles `PState`s, mirroring the orbital
# evaluation API (`evaluate(::AtomicOrbitals, X)`): each point carries a position
# `x.𝐫` (a plain `SVector{3}`, in atomic units / Bohr — no `Unitful` conversion,
# unlike the GaussianBasis-path `CompiledBasis`) and a species `x.S`.

# (3, B) position matrix in `FT` from the PState positions.
function _cob_posmat(X::AbstractVector{<:PState}, ::Type{FT}) where {FT}
    P = Matrix{FT}(undef, 3, length(X))
    @inbounds for i in eachindex(X)
        r = _position(X[i])
        P[1, i] = FT(r[1]); P[2, i] = FT(r[2]); P[3, i] = FT(r[3])
    end
    return P
end

# species label → species-axis index σ ∈ 1:NZ via `basis.zlist`.
function _cob_species_index(basis::CompiledOrbitalBasis, s)
    σ = findfirst(==(s), basis.zlist)
    σ === nothing &&
        error("species $(s) not in basis species list $(basis.zlist)")
    return σ
end

# species indices for an input batch, from each `x.S`.
_cob_sidx(basis::CompiledOrbitalBasis, X::AbstractVector{<:PState}) =
        Int[ _cob_species_index(basis, x.S) for x in X ]

"""
    batch_overlap!(out, basis::CompiledOrbitalBasis, XA, XB;
                   backend=KA.get_backend(out)) -> out

Batched 2-center Cartesian-Gaussian overlap for a species-aware
[`CompiledOrbitalBasis`](@ref). `XA`/`XB` are length-`B` vectors of
`DecoratedParticles` `PState`s; each carries a position (`x.𝐫`, atomic units /
Bohr) and a species (`x.S`). For batch element `b`, the bra basis uses `XA[b]`'s
species at `XA[b]`'s position and the ket basis uses `XB[b]`. `out` is
`(nbf_total, nbf_total, B)` and sets the kernel precision.
"""
function batch_overlap!(out, basis::CompiledOrbitalBasis,
                        XA::AbstractVector{<:PState},
                        XB::AbstractVector{<:PState};
                        backend = KA.get_backend(out))
    FT = eltype(out)
    B = length(XA)
    length(XB) == B ||
        error("XA and XB must have equal length; got $(length(XA)) and $(length(XB))")
    ArrayCtor = typeof(out).name.wrapper
    posA_d = ArrayCtor(_cob_posmat(XA, FT))
    posB_d = ArrayCtor(_cob_posmat(XB, FT))
    coef_d = ArrayCtor(FT.(basis.coef))
    α_d    = ArrayCtor(FT.(basis.ζ))
    sA_d   = ArrayCtor(_cob_sidx(basis, XA))
    sB_d   = ArrayCtor(_cob_sidx(basis, XB))
    fill!(out, zero(FT))
    kernel! = batch_S_orbital_kernel!(backend)
    kernel!(out, basis.ls, basis.nbf, basis.basis_offset, coef_d, α_d, sA_d, sB_d,
            posA_d, posB_d, Val(basis.Lmax);
            ndrange = (B, basis.nshells, basis.nshells))
    KA.synchronize(backend)
    return out
end

"""
    batch_overlap(basis::CompiledOrbitalBasis, XA, XB; FT=Float64) -> Array{FT,3}

Non-mutating wrapper that allocates the `(nbf_total, nbf_total, B)` output on the
CPU and fills it via [`batch_overlap!`](@ref). `XA`/`XB` are vectors of `PState`s.
"""
function batch_overlap(basis::CompiledOrbitalBasis,
                       XA::AbstractVector{<:PState},
                       XB::AbstractVector{<:PState};
                       FT::Type = Float64)
    N = basis.nbf_total
    out = zeros(FT, N, N, length(XA))
    return batch_overlap!(out, basis, XA, XB)
end
