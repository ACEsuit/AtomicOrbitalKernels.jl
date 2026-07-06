# Batched 3-center Cartesian-Gaussian overlap kernel.
#
# V_{μνλ}(b) = ∫ φ_μ(r - A_b) φ_ν(r - B_b) φ_λ(r - C_b) dr
#
# This is the triple-product overlap of three Gaussian basis functions at three
# distinct centers — an extension that has no direct equivalent in
# GaussianBasis.jl (its `ERI_2e3c` is a different 3-center 2-electron integral).
#
# Reuses `CompiledBasis` (same basis SoA; only the kernel differs). One
# work-item per (b, s_a, s_b, s_c).

@kernel function batch_V_kernel!(out::AbstractArray{FT,4},
                                 @Const(ls), @Const(nprim), @Const(prim_offset),
                                 @Const(coef), @Const(α),
                                 @Const(nbf), @Const(basis_offset),
                                 @Const(posA), @Const(posB), @Const(posC),
                                 ::Val{Lmax}) where {FT,Lmax}
    b, s_a, s_b, s_c = @index(Global, NTuple)

    l_a = ls[s_a];     l_b = ls[s_b];     l_c = ls[s_c]
    nbf_a = nbf[s_a];  nbf_b = nbf[s_b];  nbf_c = nbf[s_c]
    row_off  = basis_offset[s_a]
    col_off  = basis_offset[s_b]
    page_off = basis_offset[s_c]
    N1 = nbf_a
    N2 = nbf_b

    E3  = MArray{Tuple{3 * Lmax + 3, Lmax + 1, Lmax + 1, Lmax + 1, 3}, FT}(undef)
    NBF = (Lmax + 1) * (Lmax + 2) ÷ 2
    blk = MArray{Tuple{NBF * NBF * NBF}, FT}(undef)
    Tmax = 3 * Lmax + 3

    half  = FT(0.5)
    π_FT  = FT(π)
    pow15 = FT(1.5)

    @inbounds for k in 1:nbf_a*nbf_b*nbf_c
        blk[k] = zero(FT)
    end

    Ax = posA[1, b]; Ay = posA[2, b]; Az = posA[3, b]
    Bx = posB[1, b]; By = posB[2, b]; Bz = posB[3, b]
    Cx = posC[1, b]; Cy = posC[2, b]; Cz = posC[3, b]

    off_a = prim_offset[s_a]; npa = nprim[s_a]
    off_b = prim_offset[s_b]; npb = nprim[s_b]
    off_c = prim_offset[s_c]; npc = nprim[s_c]

    @inbounds for ip in 1:npa, jp in 1:npb, kp in 1:npc
        ca = coef[off_a + ip]; αa = α[off_a + ip]
        cb = coef[off_b + jp]; αb = α[off_b + jp]
        cc = coef[off_c + kp]; αc = α[off_c + kp]

        p = αa + αb
        Px = (αa * Ax + αb * Bx) / p
        Py = (αa * Ay + αb * By) / p
        Pz = (αa * Az + αb * Bz) / p
        q = p + αc
        Qx = (p * Px + αc * Cx) / q
        Qy = (p * Py + αc * Cy) / q
        Qz = (p * Pz + αc * Cz) / q
        μ  = αa * αb / p
        ν  = p  * αc / q
        oo2q = half / q
        prefac = ca * cb * cc * (π_FT / q)^pow15

        for ax in 1:3
            P_ax = ax == 1 ? Px : (ax == 2 ? Py : Pz)
            Q_ax = ax == 1 ? Qx : (ax == 2 ? Qy : Qz)
            A_ax = ax == 1 ? Ax : (ax == 2 ? Ay : Az)
            B_ax = ax == 1 ? Bx : (ax == 2 ? By : Bz)
            C_ax = ax == 1 ? Cx : (ax == 2 ? Cy : Cz)
            AB = A_ax - B_ax
            PC = P_ax - C_ax
            QA = Q_ax - A_ax
            QB = Q_ax - B_ax
            QC = Q_ax - C_ax

            for kc in 1:(l_c + 1), jb in 1:(l_b + 1), ia in 1:(l_a + 1)
                abc = ia + jb + kc - 1
                if abc <= Tmax
                    E3[abc, ia, jb, kc, ax] = zero(FT)
                end
                if abc + 1 <= Tmax
                    E3[abc+1, ia, jb, kc, ax] = zero(FT)
                end
            end
            E3[1, 1, 1, 1, ax] = exp(-μ * AB * AB - ν * PC * PC)

            for i in 2:(l_a + 1)
                E3[1, i, 1, 1, ax] = QA * E3[1, i-1, 1, 1, ax] + E3[2, i-1, 1, 1, ax]
                for t in 2:i
                    tFT = FT(t)
                    E3[t, i, 1, 1, ax] = QA * E3[t, i-1, 1, 1, ax] + tFT * E3[t+1, i-1, 1, 1, ax] + oo2q * E3[t-1, i-1, 1, 1, ax]
                end
            end
            for j in 2:(l_b + 1)
                E3[1, 1, j, 1, ax] = QB * E3[1, 1, j-1, 1, ax] + E3[2, 1, j-1, 1, ax]
                for t in 2:j
                    tFT = FT(t)
                    E3[t, 1, j, 1, ax] = QB * E3[t, 1, j-1, 1, ax] + tFT * E3[t+1, 1, j-1, 1, ax] + oo2q * E3[t-1, 1, j-1, 1, ax]
                end
                for i in 2:(l_a + 1)
                    E3[1, i, j, 1, ax] = QA * E3[1, i-1, j, 1, ax] + E3[2, i-1, j, 1, ax]
                    for t in 2:i+j-1
                        tFT = FT(t)
                        E3[t, i, j, 1, ax] = QA * E3[t, i-1, j, 1, ax] + tFT * E3[t+1, i-1, j, 1, ax] + oo2q * E3[t-1, i-1, j, 1, ax]
                    end
                end
            end
            for k in 2:(l_c + 1)
                E3[1, 1, 1, k, ax] = QC * E3[1, 1, 1, k-1, ax] + E3[2, 1, 1, k-1, ax]
                for t in 2:k
                    tFT = FT(t)
                    E3[t, 1, 1, k, ax] = QC * E3[t, 1, 1, k-1, ax] + tFT * E3[t+1, 1, 1, k-1, ax] + oo2q * E3[t-1, 1, 1, k-1, ax]
                end
                for j in 2:(l_b + 1)
                    E3[1, 1, j, k, ax] = QB * E3[1, 1, j-1, k, ax] + E3[2, 1, j-1, k, ax]
                    for t in 2:j+k-1
                        tFT = FT(t)
                        E3[t, 1, j, k, ax] = QB * E3[t, 1, j-1, k, ax] + tFT * E3[t+1, 1, j-1, k, ax] + oo2q * E3[t-1, 1, j-1, k, ax]
                    end
                end
                for j in 1:(l_b + 1), i in 2:(l_a + 1)
                    E3[1, i, j, k, ax] = QA * E3[1, i-1, j, k, ax] + E3[2, i-1, j, k, ax]
                    for t in 2:i+j+k-1
                        tFT = FT(t)
                        E3[t, i, j, k, ax] = QA * E3[t, i-1, j, k, ax] + tFT * E3[t+1, i-1, j, k, ax] + oo2q * E3[t-1, i-1, j, k, ax]
                    end
                end
            end
        end

        # Contraction: flat (nbf_a × nbf_b × nbf_c) block at linear position
        # `index1 + N1*(index2-1) + N1*N2*(index3-1)`, N1 = nbf_a, N2 = nbf_b (the
        # Cartesian counts), matching the readback strides below. Matches
        # Reference.generate_V_triple! bit-for-bit.
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
                    blk[lin] += E3[1, ll1+1, ll2+1, ll3+1, 1] *
                                E3[1, m1+1,  m2+1,  m3+1,  2] *
                                E3[1, n1+1,  n2+1,  n3+1,  3] * prefac
                    index3 += 1
                end
                index2 += 1
            end
            index1 += 1
        end
    end

    @inbounds for kk in 1:nbf_c, jj in 1:nbf_b, ii in 1:nbf_a
        lin = ii + nbf_a * (jj - 1) + nbf_a * nbf_b * (kk - 1)
        out[row_off + ii, col_off + jj, page_off + kk, b] = blk[lin]
    end
end

"""
    batch_overlap_3c!(out, basis::CompiledBasis, posA, posB, posC; backend=KA.get_backend(out)) -> out

Compute the batched 3-center Cartesian-Gaussian overlap integrals
``V_{μνλ}(b) = ∫ φ_μ(\\mathbf{r} - \\mathbf{A}_b) \\, φ_ν(\\mathbf{r} - \\mathbf{B}_b) \\, φ_λ(\\mathbf{r} - \\mathbf{C}_b) \\, \\mathrm{d}\\mathbf{r}``
for each batch element `b ∈ 1:B`. This integral has no direct equivalent in
GaussianBasis.jl (its `ERI_2e3c` is a different 2-electron 3-center integral).

Arguments:

- `out`     : `(N, N, N, B)` array, `N = basis.nbf_total`. Element type sets
              the kernel's float precision.
- `basis`   : a [`CompiledBasis`](@ref) (the same struct used by 2C).
- `posA`, `posB`, `posC` : `(3, B)` matrices of `Unitful.Length`.
- `backend` : KernelAbstractions backend; inferred from `out`.

The output tensor scales as ``O(N^3 B)``, so 3C runs typically use a much
smaller `B` than 2C.
"""
function batch_overlap_3c!(out, basis::CompiledBasis,
                           posA::AbstractMatrix{<:Unitful.Length},
                           posB::AbstractMatrix{<:Unitful.Length},
                           posC::AbstractMatrix{<:Unitful.Length};
                           backend = KA.get_backend(out))
    FT = eltype(out)
    ArrayCtor = typeof(out).name.wrapper
    posA_d = ArrayCtor(to_bohr(posA, FT))
    posB_d = ArrayCtor(to_bohr(posB, FT))
    posC_d = ArrayCtor(to_bohr(posC, FT))
    fill!(out, zero(FT))
    kernel! = batch_V_kernel!(backend)
    kernel!(out, basis.ls, basis.nprim, basis.prim_offset, basis.coef, basis.α,
            basis.nbf, basis.basis_offset, posA_d, posB_d, posC_d, Val(basis.Lmax);
            ndrange = (size(posA, 2), basis.nshells, basis.nshells, basis.nshells))
    KA.synchronize(backend)
    return out
end

batch_overlap_3c!(out, basis::CompiledBasis,
                  posA::AbstractMatrix, posB::AbstractMatrix, posC::AbstractMatrix;
                  kwargs...) =
    throw(ArgumentError(_UNITFUL_HINT))

"""
    batch_overlap_3c(basis::CompiledBasis, posA, posB, posC; FT=Float64) -> Array{FT,4}

Non-mutating convenience wrapper that allocates `(N, N, N, B)` on the CPU and
fills it via [`batch_overlap_3c!`](@ref).
"""
function batch_overlap_3c(basis::CompiledBasis,
                          posA::AbstractMatrix{<:Unitful.Length},
                          posB::AbstractMatrix{<:Unitful.Length},
                          posC::AbstractMatrix{<:Unitful.Length};
                          FT::Type = Float64)
    B = size(posA, 2)
    N = basis.nbf_total
    out = zeros(FT, N, N, N, B)
    return batch_overlap_3c!(out, basis, posA, posB, posC)
end
