# Batched 2-center Cartesian-Gaussian overlap for a species-aware
# `CartesianGTOBasis`. Mirrors `batch_S_kernel!` (same McMurchie–Davidson
# E-recursion + contraction + block write, including the `N1 = nbf_a` linear
# stride) but reads per-species primitives: the bra shells use species `sidxA[b]`
# and the ket shells use species `sidxB[b]`, so each batch element is an overlap
# between an atom of one species and an atom of another.
#
# Primitive slots are uniform (`K`, padded with `coef=0`); zero-coef pairs are
# skipped. The shared `(n,l)` spec makes every species' shell structure identical,
# so the output block is `nbf_total × nbf_total` for any species pair.
#
# Recursion + contraction come from the shared helpers in `integrals/mdE.jl`;
# only the per-species primitive access and the zero-coef skip live here.

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
    N1 = nbf_a
    K = size(coef, 2)

    E   = MArray{Tuple{2 * Lmax + 2, Lmax + 1, Lmax + 1, 3}, FT}(undef)
    blk = MArray{Tuple{((Lmax + 1) * (Lmax + 2) ÷ 2)^2}, FT}(undef)

    half_oo = FT(0.5)
    π_FT = FT(π)
    pow15 = FT(1.5)

    @inbounds for k in 1:nbf_a*nbf_b
        blk[k] = zero(FT)
    end

    A = SVector(posA[1, b], posA[2, b], posA[3, b])
    B = SVector(posB[1, b], posB[2, b], posB[3, b])

    @inbounds for ip in 1:K, jp in 1:K
        ca = coef[s_a, ip, σa]
        cb = coef[s_b, jp, σb]
        (ca == zero(FT) || cb == zero(FT)) && continue
        αa = α[s_a, ip, σa]
        αb = α[s_b, jp, σb]

        p, P, μ = _gaussprod_2c(αa, αb, A, B)
        oo2p = half_oo / p
        prefac = ca * cb * (π_FT / p)^pow15

        _mdE_2c!(E, l_a + 1, l_b + 1, A, B, P, μ, oo2p)
        _contract_S_2c!(blk, E, l_a, l_b, N1, prefac)
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
function _cgto_posmat(X::AbstractVector{<:PState}, ::Type{FT}) where {FT}
    P = Matrix{FT}(undef, 3, length(X))
    @inbounds for i in eachindex(X)
        r = _position(X[i])
        P[1, i] = FT(r[1]); P[2, i] = FT(r[2]); P[3, i] = FT(r[3])
    end
    return P
end

# species label → species-axis index σ ∈ 1:NZ via `basis.zlist`.
function _cgto_species_index(basis::CartesianGTOBasis, s)
    σ = findfirst(==(s), basis.zlist)
    σ === nothing &&
        error("species $(s) not in basis species list $(basis.zlist)")
    return σ
end

# species indices for an input batch, from each `x.S`.
_cgto_sidx(basis::CartesianGTOBasis, X::AbstractVector{<:PState}) =
        Int[ _cgto_species_index(basis, x.S) for x in X ]

"""
    batch_overlap!(out, basis::CartesianGTOBasis, XA, XB;
                   backend=KA.get_backend(out)) -> out

Batched 2-center Cartesian-Gaussian overlap for a species-aware
[`CartesianGTOBasis`](@ref). `XA`/`XB` are length-`B` vectors of
`DecoratedParticles` `PState`s; each carries a position (`x.𝐫`, in the basis's
`length_unit`) and a species (`x.S`). For batch element `b`, the bra basis uses `XA[b]`'s
species at `XA[b]`'s position and the ket basis uses `XB[b]`. `out` is
`(nbf_total, nbf_total, B)` and sets the kernel precision.
"""
function batch_overlap!(out, basis::CartesianGTOBasis,
                        XA::AbstractVector{<:PState},
                        XB::AbstractVector{<:PState};
                        backend = KA.get_backend(out))
    FT = eltype(out)
    B = length(XA)
    length(XB) == B ||
        error("XA and XB must have equal length; got $(length(XA)) and $(length(XB))")
    ArrayCtor = typeof(out).name.wrapper
    # scale input positions into the native (Bohr) frame of the parameters
    s = FT(basis.lengthscale)
    posA = _cgto_posmat(XA, FT);  s == one(s) || (posA .*= s)
    posB = _cgto_posmat(XB, FT);  s == one(s) || (posB .*= s)
    posA_d = ArrayCtor(posA)
    posB_d = ArrayCtor(posB)
    coef_d = ArrayCtor(FT.(basis.coef))
    α_d    = ArrayCtor(FT.(basis.ζ))
    sA_d   = ArrayCtor(_cgto_sidx(basis, XA))
    sB_d   = ArrayCtor(_cgto_sidx(basis, XB))
    fill!(out, zero(FT))
    kernel! = batch_S_orbital_kernel!(backend)
    kernel!(out, basis.ls, basis.nbf, basis.basis_offset, coef_d, α_d, sA_d, sB_d,
            posA_d, posB_d, Val(basis.Lmax);
            ndrange = (B, basis.nshells, basis.nshells))
    KA.synchronize(backend)
    return out
end

"""
    batch_overlap(basis::CartesianGTOBasis, XA, XB) -> Array

Non-mutating wrapper that allocates the `(nbf_total, nbf_total, B)` output on the
CPU and fills it via [`batch_overlap!`](@ref). `XA`/`XB` are vectors of `PState`s;
the output precision is taken from the input positions' element type.
"""
function batch_overlap(basis::CartesianGTOBasis,
                       XA::AbstractVector{<:PState},
                       XB::AbstractVector{<:PState})
    FT = float(eltype(_position(first(XA))))
    N = basis.nbf_total
    out = zeros(FT, N, N, length(XA))
    return batch_overlap!(out, basis, XA, XB)
end

# Convenience overloads on the orbital basis itself: compile to a
# `CartesianGTOBasis` on the fly, then run the kernel. Handy for training (the
# compile is cheap and — once Stage 4 lands — differentiable in `(ζ,D)`); for
# inference, compile once with `compile_basis` and reuse the result.

"""
    batch_overlap(orb::AtomicOrbitals, XA, XB) -> Array
    batch_overlap!(out, orb::AtomicOrbitals, XA, XB) -> out

Compile the Gaussian orbital basis `orb` to a [`CartesianGTOBasis`](@ref) and
compute the 2-center overlap. Equivalent to
`batch_overlap[!](compile_basis(orb), XA, XB)`.
"""
batch_overlap(orb::GaussianTypeOrbitals,
              XA::AbstractVector{<:PState}, XB::AbstractVector{<:PState}) =
        batch_overlap(compile_basis(orb), XA, XB)

batch_overlap!(out, orb::GaussianTypeOrbitals,
               XA::AbstractVector{<:PState}, XB::AbstractVector{<:PState};
               backend = KA.get_backend(out)) =
        batch_overlap!(out, compile_basis(orb), XA, XB; backend = backend)
