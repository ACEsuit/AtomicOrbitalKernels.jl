# Batched 2-center Cartesian-Gaussian overlap kernel.
#
# One work-item per (batch_idx, shell_a, shell_b). Each does the full
# E-coefficient recursion (McMurchie–Davidson / Obara–Saika) plus contraction
# using per-work-item stack-allocated scratch sized at compile time by
# `Val{Lmax}`. The xyz axis is the slowest scratch dimension so the t-loop is
# stride-1 per axis (enables `@simd` on CPU; no effect on GPU).
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
    N1 = 2 * l_a + 1

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

    off_a = prim_offset[s_a]; npa = nprim[s_a]
    off_b = prim_offset[s_b]; npb = nprim[s_b]

    @inbounds for ip in 1:npa, jp in 1:npb
        ca = coef[off_a + ip]; αa = α[off_a + ip]
        cb = coef[off_b + jp]; αb = α[off_b + jp]

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

        # Contraction: flat (nbf_a × nbf_b) block at linear position
        # `index1 + N1*(index2-1)`, N1 = 2*l_a + 1. Matches Reference.generate_S_pair!
        # bit-for-bit (including the L ≥ 2 aliasing).
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

