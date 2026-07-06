# Batched 3-center Cartesian-Gaussian overlap for a species-aware
# `CartesianGTOBasis`. Mirrors `batch_V_kernel!` (same E3 recursion +
# contraction + block write, including the `N1/N2` linear strides) but reads
# per-species primitives: the three shells use species `sidxA[b]`, `sidxB[b]`
# and `sidxC[b]`, so each batch element is a triple overlap between atoms of
# (possibly) three different species.
#
# Primitive slots are uniform (`K`, padded with `coef=0`); zero-coef triples
# are skipped. The shared `(n,l)` spec makes every species' shell structure
# identical, so the output block is `nbf_total³` for any species triple.
#
# Recursion + contraction come from the shared helpers in `integrals/mdE.jl`;
# only the per-species primitive access and the zero-coef skip live here.
# Forward only — the `(ζ,D)` pullback currently exists for the 2C path alone
# (`overlap_2c_grad.jl`).

@kernel function batch_V_orbital_kernel!(out::AbstractArray{FT,4},
                                 @Const(ls), @Const(nbf), @Const(basis_offset),
                                 @Const(coef), @Const(α),
                                 @Const(sidxA), @Const(sidxB), @Const(sidxC),
                                 @Const(posA), @Const(posB), @Const(posC),
                                 ::Val{Lmax}) where {FT,Lmax}
    b, s_a, s_b, s_c = @index(Global, NTuple)

    σa = sidxA[b]
    σb = sidxB[b]
    σc = sidxC[b]
    l_a = ls[s_a];     l_b = ls[s_b];     l_c = ls[s_c]
    nbf_a = nbf[s_a];  nbf_b = nbf[s_b];  nbf_c = nbf[s_c]
    row_off  = basis_offset[s_a]
    col_off  = basis_offset[s_b]
    page_off = basis_offset[s_c]
    N1 = nbf_a
    N2 = nbf_b
    K = size(coef, 2)

    E3  = MArray{Tuple{3 * Lmax + 3, Lmax + 1, Lmax + 1, Lmax + 1, 3}, FT}(undef)
    NBF = (Lmax + 1) * (Lmax + 2) ÷ 2
    blk = MArray{Tuple{NBF * NBF * NBF}, FT}(undef)

    half_oo = FT(0.5)
    π_FT = FT(π)
    pow15 = FT(1.5)

    @inbounds for k in 1:nbf_a*nbf_b*nbf_c
        blk[k] = zero(FT)
    end

    A = SVector(posA[1, b], posA[2, b], posA[3, b])
    B = SVector(posB[1, b], posB[2, b], posB[3, b])
    C = SVector(posC[1, b], posC[2, b], posC[3, b])

    @inbounds for ip in 1:K, jp in 1:K, kp in 1:K
        ca = coef[s_a, ip, σa]
        cb = coef[s_b, jp, σb]
        cc = coef[s_c, kp, σc]
        (ca == zero(FT) || cb == zero(FT) || cc == zero(FT)) && continue
        αa = α[s_a, ip, σa]
        αb = α[s_b, jp, σb]
        αc = α[s_c, kp, σc]

        q, Q, P, μ, ν = _gaussprod_3c(αa, αb, αc, A, B, C)
        oo2q = half_oo / q
        prefac = ca * cb * cc * (π_FT / q)^pow15

        _mdE_3c!(E3, l_a + 1, l_b + 1, l_c + 1, A, B, C, P, Q, μ, ν, oo2q)
        _contract_V_3c!(blk, E3, l_a, l_b, l_c, N1, N2, prefac)
    end

    @inbounds for kk in 1:nbf_c, jj in 1:nbf_b, ii in 1:nbf_a
        lin = ii + nbf_a * (jj - 1) + nbf_a * nbf_b * (kk - 1)
        out[row_off + ii, col_off + jj, page_off + kk, b] = blk[lin]
    end
end

"""
    batch_overlap!(out, basis::CartesianGTOBasis, XA, XB, XC;
                   backend=KA.get_backend(out)) -> out

Batched 3-center Cartesian-Gaussian overlap for a species-aware
[`CartesianGTOBasis`](@ref). `XA`/`XB`/`XC` are length-`B` vectors of
`DecoratedParticles` `PState`s; each carries a position (`x.𝐫`, in the basis's
`length_unit`) and a species (`x.S`). For batch element `b`, the three basis
sets use `XA[b]`, `XB[b]` and `XC[b]` respectively. `out` is
`(nbf_total, nbf_total, nbf_total, B)` and sets the kernel precision. The
output scales as ``O(N^3 B)``, so 3C runs typically use a much smaller `B`
than 2C.
"""
function batch_overlap!(out, basis::CartesianGTOBasis,
                        XA::AbstractVector{<:PState},
                        XB::AbstractVector{<:PState},
                        XC::AbstractVector{<:PState};
                        backend = KA.get_backend(out))
    FT = eltype(out)
    B = length(XA)
    length(XB) == B && length(XC) == B ||
        error("XA, XB and XC must have equal length; got $(length(XA)), \
              $(length(XB)) and $(length(XC))")
    ArrayCtor = typeof(out).name.wrapper
    # scale input positions into the native (Bohr) frame of the parameters
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
    fill!(out, zero(FT))
    kernel! = batch_V_orbital_kernel!(backend)
    kernel!(out, basis.ls, basis.nbf, basis.basis_offset, coef_d, α_d,
            sA_d, sB_d, sC_d, posA_d, posB_d, posC_d, Val(basis.Lmax);
            ndrange = (B, basis.nshells, basis.nshells, basis.nshells))
    KA.synchronize(backend)
    return out
end

"""
    batch_overlap(basis::CartesianGTOBasis, XA, XB, XC) -> Array

Non-mutating wrapper that allocates the `(nbf_total, nbf_total, nbf_total, B)`
output on the CPU and fills it via [`batch_overlap!`](@ref). `XA`/`XB`/`XC`
are vectors of `PState`s; the output precision is taken from the input
positions' element type.
"""
function batch_overlap(basis::CartesianGTOBasis,
                       XA::AbstractVector{<:PState},
                       XB::AbstractVector{<:PState},
                       XC::AbstractVector{<:PState})
    FT = float(eltype(_position(first(XA))))
    N = basis.nbf_total
    out = zeros(FT, N, N, N, length(XA))
    return batch_overlap!(out, basis, XA, XB, XC)
end

"""
    batch_overlap(orb::AtomicOrbitals, XA, XB, XC) -> Array
    batch_overlap!(out, orb::AtomicOrbitals, XA, XB, XC) -> out

Compile the Gaussian orbital basis `orb` to a [`CartesianGTOBasis`](@ref) and
compute the 3-center overlap. Equivalent to
`batch_overlap[!](compile_basis(orb), XA, XB, XC)`.
"""
batch_overlap(orb::GaussianTypeOrbitals,
              XA::AbstractVector{<:PState}, XB::AbstractVector{<:PState},
              XC::AbstractVector{<:PState}) =
        batch_overlap(compile_basis(orb), XA, XB, XC)

batch_overlap!(out, orb::GaussianTypeOrbitals,
               XA::AbstractVector{<:PState}, XB::AbstractVector{<:PState},
               XC::AbstractVector{<:PState};
               backend = KA.get_backend(out)) =
        batch_overlap!(out, compile_basis(orb), XA, XB, XC; backend = backend)
