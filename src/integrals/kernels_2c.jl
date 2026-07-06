# Batched 2-center Cartesian-Gaussian overlap kernel.
#
# One work-item per (batch_idx, shell_a, shell_b). Each does the full
# E-coefficient recursion (McMurchie–Davidson / Obara–Saika) plus contraction
# using per-work-item stack-allocated scratch sized at compile time by
# `Val{Lmax}`. The recursion and contraction live in the shared helpers
# (`integrals/mdE.jl`), which this kernel drives with `imax/jmax = l+1`.
#
# Positions arrive in Bohr (units stripped + converted at the wrapper).

@kernel function batch_S_kernel!(out::AbstractArray{FT,3},
                                 @Const(ls), @Const(nprim), @Const(prim_offset),
                                 @Const(coef), @Const(α),
                                 @Const(nbf), @Const(basis_offset),
                                 @Const(posA), @Const(posB),
                                 ::Val{Lmax}) where {FT,Lmax}
    b, s_a, s_b = @index(Global, NTuple)

    l_a = ls[s_a]
    l_b = ls[s_b]
    nbf_a = nbf[s_a]
    nbf_b = nbf[s_b]
    row_off = basis_offset[s_a]
    col_off = basis_offset[s_b]
    N1 = nbf_a

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

    off_a = prim_offset[s_a]; npa = nprim[s_a]
    off_b = prim_offset[s_b]; npb = nprim[s_b]

    @inbounds for ip in 1:npa, jp in 1:npb
        ca = coef[off_a + ip]; αa = α[off_a + ip]
        cb = coef[off_b + jp]; αb = α[off_b + jp]

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

"""
    batch_overlap!(out, basis::CompiledBasis, posA, posB; backend=KA.get_backend(out)) -> out

Compute the batched 2-center Cartesian-Gaussian overlap integrals
``S_{μν}(b) = ⟨φ_μ(\\mathbf{r} - \\mathbf{A}_b) \\,|\\, φ_ν(\\mathbf{r} - \\mathbf{B}_b)⟩``
for each batch element `b ∈ 1:B`, writing into `out`.

Arguments:

- `out`     : `(N, N, B)` array, where `N = basis.nbf_total`. Element type
              determines the kernel's float precision.
- `basis`   : a [`CompiledBasis`](@ref) obtained from [`compile_basis`](@ref)
              (optionally moved to a device via [`adapt_basis`](@ref)).
- `posA`, `posB` : `(3, B)` matrices of `Unitful.Length` positions. Plain
              `Real` matrices are rejected — use Unitful to make units explicit.
- `backend` : KernelAbstractions backend; inferred from `out` by default.

The backend is selected by the type of `out` (`Array` → CPU; `CuArray` → CUDA;
`MtlArray` → Metal; etc.). For GPU runs, move both the basis and positions to
the same device before calling, and prefer `Float32` precision.
"""
function batch_overlap!(out, basis::CompiledBasis,
                        posA::AbstractMatrix{<:Unitful.Length},
                        posB::AbstractMatrix{<:Unitful.Length};
                        backend = KA.get_backend(out))
    FT = eltype(out)
    ArrayCtor = typeof(out).name.wrapper
    posA_d = ArrayCtor(to_bohr(posA, FT))
    posB_d = ArrayCtor(to_bohr(posB, FT))
    fill!(out, zero(FT))
    kernel! = batch_S_kernel!(backend)
    kernel!(out, basis.ls, basis.nprim, basis.prim_offset, basis.coef, basis.α,
            basis.nbf, basis.basis_offset, posA_d, posB_d, Val(basis.Lmax);
            ndrange = (size(posA, 2), basis.nshells, basis.nshells))
    KA.synchronize(backend)
    return out
end

batch_overlap!(out, basis::CompiledBasis,
               posA::AbstractMatrix, posB::AbstractMatrix; kwargs...) =
    throw(ArgumentError(_UNITFUL_HINT))

"""
    batch_overlap(basis::CompiledBasis, posA, posB; FT=Float64) -> Array{FT,3}

Non-mutating convenience wrapper that allocates the `(N, N, B)` output on the
CPU and fills it via [`batch_overlap!`](@ref). For GPU runs you should
preallocate `out` on the device and call `batch_overlap!` directly.
"""
function batch_overlap(basis::CompiledBasis,
                       posA::AbstractMatrix{<:Unitful.Length},
                       posB::AbstractMatrix{<:Unitful.Length};
                       FT::Type = Float64)
    B = size(posA, 2)
    N = basis.nbf_total
    out = zeros(FT, N, N, B)
    return batch_overlap!(out, basis, posA, posB)
end

