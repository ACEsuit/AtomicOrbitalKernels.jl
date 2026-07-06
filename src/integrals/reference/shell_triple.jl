"""
    generate_V_triple!(out, E3, B1, B2, B3)

Generate the 3-center overlap block for the shell triple `(B1, B2, B3)`. Same
recursion family as the 2-center case but with two Gaussian-product reductions:
`(B1, B2) → P` first, then `(P, B3) → Q`. Linear write at
`index1 + N1*(index2-1) + N1*N2*(index3-1)`.
"""
function generate_V_triple!(out, E3, B1, B2, B3)
    fill!(out, zero(eltype(out)))

    am1 = B1.l
    am2 = B2.l
    am3 = B3.l

    A = B1.atom.xyz .* ang2bohr
    B = B2.atom.xyz .* ang2bohr
    C = B3.atom.xyz .* ang2bohr

    N1 = (B1.l + 1) * (B1.l + 2) ÷ 2
    N2 = (B2.l + 1) * (B2.l + 2) ÷ 2

    for (ca, a) in zip(B1.coef, B1.exp)
        for (cb, b) in zip(B2.coef, B2.exp)
            for (cc, c) in zip(B3.coef, B3.exp)
                p = a + b
                P = (a * A + b * B) / p
                q = p + c
                Q = (p * P + c * C) / q
                prefac = ca * cb * cc * (π / q)^1.5

                generate_E3_matrix!(E3, am1, am2, am3, P, Q, A, B, C, a, b, c)

                index1 = 1
                for l1 in am1:-1:0, n1 in 0:am1-l1
                    m1 = am1 - l1 - n1
                    index2 = 1
                    for l2 in am2:-1:0, n2 in 0:am2-l2
                        m2 = am2 - l2 - n2
                        index3 = 1
                        for l3 in am3:-1:0, n3 in 0:am3-l3
                            m3 = am3 - l3 - n3
                            index = index1 + N1 * (index2 - 1) + N1 * N2 * (index3 - 1)
                            out[index] +=
                                E3[1, 1, l1+1, l2+1, l3+1] *
                                E3[2, 1, m1+1, m2+1, m3+1] *
                                E3[3, 1, n1+1, n2+1, n3+1] * prefac
                            index3 += 1
                        end
                        index2 += 1
                    end
                    index1 += 1
                end
            end
        end
    end
    return out
end

"""
    batch_V_triple_ref!(out, BS, posA, posB, posC)

Naive reference for the batched 3-center overlap. For each batch element `b`,
compute the full `N × N × N` block tensor between basis functions centered at
`posA[:, b]`, `posB[:, b]`, `posC[:, b]` (Å, plain numeric matrices). Used as
the correctness oracle for `batch_overlap_3c!`.
"""
function batch_V_triple_ref!(out::AbstractArray{T,4}, BS,
                             posA::AbstractMatrix, posB::AbstractMatrix,
                             posC::AbstractMatrix) where {T}
    nshells = length(BS.basis)
    nbf = Vector{Int}(undef, nshells)
    for i in 1:nshells
        l = BS.basis[i].l
        nbf[i] = (l + 1) * (l + 2) ÷ 2
    end
    offset = Vector{Int}(undef, nshells + 1)
    offset[1] = 0
    for i in 1:nshells
        offset[i+1] = offset[i] + nbf[i]
    end
    Ntot = offset[end]
    @assert size(out, 1) == Ntot && size(out, 2) == Ntot && size(out, 3) == Ntot

    fill!(out, zero(T))
    for b in 1:size(posA, 2)
        𝐫A = SVector{3,Float64}(posA[1, b], posA[2, b], posA[3, b])
        𝐫B = SVector{3,Float64}(posB[1, b], posB[2, b], posB[3, b])
        𝐫C = SVector{3,Float64}(posC[1, b], posC[2, b], posC[3, b])
        for s_a in 1:nshells, s_b in 1:nshells, s_c in 1:nshells
            sa = BS.basis[s_a]
            sb = BS.basis[s_b]
            sc = BS.basis[s_c]
            new_sa = CartesianShell(sa.l, sa.coef, sa.exp,
                                    Molecules.Atom(sa.atom.Z, sa.atom.mass, 𝐫A))
            new_sb = CartesianShell(sb.l, sb.coef, sb.exp,
                                    Molecules.Atom(sb.atom.Z, sb.atom.mass, 𝐫B))
            new_sc = CartesianShell(sc.l, sc.coef, sc.exp,
                                    Molecules.Atom(sc.atom.Z, sc.atom.mass, 𝐫C))
            block = view(out, offset[s_a]+1:offset[s_a]+nbf[s_a],
                              offset[s_b]+1:offset[s_b]+nbf[s_b],
                              offset[s_c]+1:offset[s_c]+nbf[s_c], b)
            E3 = alloc_E3(new_sa, new_sb, new_sc)
            generate_V_triple!(block, E3, new_sa, new_sb, new_sc)
        end
    end
    return out
end
