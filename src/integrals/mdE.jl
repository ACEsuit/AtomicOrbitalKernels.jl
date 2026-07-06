# Shared McMurchie–Davidson E-coefficient recursion for the Cartesian-Gaussian
# overlap kernels, plus the Gaussian-product and contraction helpers around it.
# All are `@inline` device functions that mutate caller-supplied compile-time
# sized `MArray` scratch — no allocation inside, KA-legal on CPU and GPU.
#
# Used by `batch_S_kernel!` (kernels_2c.jl), `batch_S_orbital_kernel!`
# (orbital/overlap_2c.jl), `batch_S_orbital_pb_kernel!`
# (orbital/overlap_2c_grad.jl — same recursion with raised bounds `l+3` for the
# ∂/∂ζ tables) and `batch_V_kernel!` (kernels_3c.jl). The scalar `Reference`
# submodule keeps its own axis-first CPU implementation (reference/ecoeffs.jl)
# on purpose — it is the independent cross-check, not a call site.
#
# Positions/centers are passed as length-3 indexables (`SVector`); the xyz axis
# is the slowest scratch dimension so the t-loop is stride-1 per axis.

# Gaussian product of two primitives: total exponent `p`, product center `P`,
# reduced exponent `μ`. Component order of operations matches the previous
# in-kernel scalars bit-for-bit.
@inline function _gaussprod_2c(αa, αb, A, B)
    p = αa + αb
    P = (αa .* A .+ αb .* B) ./ p
    μ = αa * αb / p
    return p, P, μ
end

# Iterated Gaussian product of three primitives: (αa,A)+(αb,B) → (p,P), then
# (p,P)+(αc,C) → (q,Q); reduced exponents `μ = αaαb/p`, `ν = pαc/q`. `P` is
# also returned — the E3 seed needs the intermediate center.
@inline function _gaussprod_3c(αa, αb, αc, A, B, C)
    p = αa + αb
    P = (αa .* A .+ αb .* B) ./ p
    q = p + αc
    Q = (p .* P .+ αc .* C) ./ q
    μ = αa * αb / p
    ν = p * αc / q
    return q, Q, P, μ, ν
end

# 2-center E-coefficient recursion for one primitive pair, all three axes.
# Fills `E[t, i, j, ax]` for `i ≤ imax`, `j ≤ jmax` — pass `l+1` for overlap
# values, `l+3` for the raised-angular-momentum pullback tables. `E` is a
# `(Tmax, imax₊, jmax₊, 3)` `MArray` with `Tmax = size(E, 1)`; entries beyond
# the recursion's reach are left untouched.
@inline function _mdE_2c!(E, imax, jmax, A, B, P, μ, oo2p)
    FT = eltype(E)
    Tmax = size(E, 1)
    @inbounds for ax in 1:3
        AB = A[ax] - B[ax]
        PA = P[ax] - A[ax]
        PB = P[ax] - B[ax]

        for jb in 1:jmax, ia in 1:imax
            ab = ia + jb
            if ab <= Tmax
                E[ab, ia, jb, ax] = zero(FT)
            end
            if ab + 1 <= Tmax
                E[ab+1, ia, jb, ax] = zero(FT)
            end
        end
        E[1, 1, 1, ax] = exp(-μ * AB * AB)

        for i in 2:imax
            E[1, i, 1, ax] = PA * E[1, i-1, 1, ax] + E[2, i-1, 1, ax]
            for t in 2:i
                tFT = FT(t)
                E[t, i, 1, ax] = PA * E[t, i-1, 1, ax] + tFT * E[t+1, i-1, 1, ax] +
                                 oo2p * E[t-1, i-1, 1, ax]
            end
        end
        for j in 2:jmax
            E[1, 1, j, ax] = PB * E[1, 1, j-1, ax] + E[2, 1, j-1, ax]
            for t in 2:j
                tFT = FT(t)
                E[t, 1, j, ax] = PB * E[t, 1, j-1, ax] + tFT * E[t+1, 1, j-1, ax] +
                                 oo2p * E[t-1, 1, j-1, ax]
            end
            for i in 2:imax
                E[1, i, j, ax] = PA * E[1, i-1, j, ax] + E[2, i-1, j, ax]
                for t in 2:i+j-1
                    tFT = FT(t)
                    E[t, i, j, ax] = PA * E[t, i-1, j, ax] + tFT * E[t+1, i-1, j, ax] +
                                     oo2p * E[t-1, i-1, j, ax]
                end
            end
        end
    end
    return nothing
end

# 3-center E-coefficient recursion for one primitive triple, all three axes.
# Same MD recursion as `_mdE_2c!` run against the combined center `Q`, with the
# third index built on top; the seed carries both Gaussian-product prefactors.
@inline function _mdE_3c!(E3, imax, jmax, kmax, A, B, C, P, Q, μ, ν, oo2q)
    FT = eltype(E3)
    Tmax = size(E3, 1)
    @inbounds for ax in 1:3
        AB = A[ax] - B[ax]
        PC = P[ax] - C[ax]
        QA = Q[ax] - A[ax]
        QB = Q[ax] - B[ax]
        QC = Q[ax] - C[ax]

        for kc in 1:kmax, jb in 1:jmax, ia in 1:imax
            abc = ia + jb + kc - 1
            if abc <= Tmax
                E3[abc, ia, jb, kc, ax] = zero(FT)
            end
            if abc + 1 <= Tmax
                E3[abc+1, ia, jb, kc, ax] = zero(FT)
            end
        end
        E3[1, 1, 1, 1, ax] = exp(-μ * AB * AB - ν * PC * PC)

        for i in 2:imax
            E3[1, i, 1, 1, ax] = QA * E3[1, i-1, 1, 1, ax] + E3[2, i-1, 1, 1, ax]
            for t in 2:i
                tFT = FT(t)
                E3[t, i, 1, 1, ax] = QA * E3[t, i-1, 1, 1, ax] +
                                     tFT * E3[t+1, i-1, 1, 1, ax] +
                                     oo2q * E3[t-1, i-1, 1, 1, ax]
            end
        end
        for j in 2:jmax
            E3[1, 1, j, 1, ax] = QB * E3[1, 1, j-1, 1, ax] + E3[2, 1, j-1, 1, ax]
            for t in 2:j
                tFT = FT(t)
                E3[t, 1, j, 1, ax] = QB * E3[t, 1, j-1, 1, ax] +
                                     tFT * E3[t+1, 1, j-1, 1, ax] +
                                     oo2q * E3[t-1, 1, j-1, 1, ax]
            end
            for i in 2:imax
                E3[1, i, j, 1, ax] = QA * E3[1, i-1, j, 1, ax] + E3[2, i-1, j, 1, ax]
                for t in 2:i+j-1
                    tFT = FT(t)
                    E3[t, i, j, 1, ax] = QA * E3[t, i-1, j, 1, ax] +
                                         tFT * E3[t+1, i-1, j, 1, ax] +
                                         oo2q * E3[t-1, i-1, j, 1, ax]
                end
            end
        end
        for k in 2:kmax
            E3[1, 1, 1, k, ax] = QC * E3[1, 1, 1, k-1, ax] + E3[2, 1, 1, k-1, ax]
            for t in 2:k
                tFT = FT(t)
                E3[t, 1, 1, k, ax] = QC * E3[t, 1, 1, k-1, ax] +
                                     tFT * E3[t+1, 1, 1, k-1, ax] +
                                     oo2q * E3[t-1, 1, 1, k-1, ax]
            end
            for j in 2:jmax
                E3[1, 1, j, k, ax] = QB * E3[1, 1, j-1, k, ax] + E3[2, 1, j-1, k, ax]
                for t in 2:j+k-1
                    tFT = FT(t)
                    E3[t, 1, j, k, ax] = QB * E3[t, 1, j-1, k, ax] +
                                         tFT * E3[t+1, 1, j-1, k, ax] +
                                         oo2q * E3[t-1, 1, j-1, k, ax]
                end
            end
            for j in 1:jmax, i in 2:imax
                E3[1, i, j, k, ax] = QA * E3[1, i-1, j, k, ax] + E3[2, i-1, j, k, ax]
                for t in 2:i+j+k-1
                    tFT = FT(t)
                    E3[t, i, j, k, ax] = QA * E3[t, i-1, j, k, ax] +
                                         tFT * E3[t+1, i-1, j, k, ax] +
                                         oo2q * E3[t-1, i-1, j, k, ax]
                end
            end
        end
    end
    return nothing
end

# Contraction of the 2-center E-tables into the flat `(nbf_a × nbf_b)` block at
# linear position `index1 + N1*(index2-1)`, `N1 = nbf_a` (the Cartesian count),
# so the write stride matches the block readback for every l. Matches
# Reference.generate_S_pair! bit-for-bit. (The pullback kernel keeps its own
# contraction — it accumulates cotangent sums, not a block.)
@inline function _contract_S_2c!(blk, E, l_a, l_b, N1, prefac)
    @inbounds begin
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
    return nothing
end
