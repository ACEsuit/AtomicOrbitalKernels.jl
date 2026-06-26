"""
    generate_S_pair!(out, E, B1, B2)

Generate the `(2l_1+1) × (2l_2+1)` block of the overlap matrix for the shell
pair `(B1, B2)` (Cartesian-Gaussian shells; positions in Å on the shell atoms).
Writes via the flat-linear `index1 + N1*(index2-1)` indexing scheme with
`N1 = 2*l_1+1` — the same convention the KA kernel reproduces so that batched
and reference results agree bit-for-bit.
"""
function generate_S_pair!(out, E, B1, B2)
    fill!(out, zero(eltype(out)))
    fill!(E, zero(eltype(E)))

    am1 = B1.l
    am2 = B2.l

    A = B1.atom.xyz .* ang2bohr
    B = B2.atom.xyz .* ang2bohr

    N1 = 2 * B1.l + 1

    for (ca, a) in zip(B1.coef, B1.exp)
        for (cb, b) in zip(B2.coef, B2.exp)
            p = a + b
            P = (a * A + b * B) / p
            prefac = ca * cb * (π / p)^1.5

            generate_E_matrix!(E, am1, am2, P, A, B, a, b)

            index1 = 1
            for l1 in am1:-1:0
                for n1 in 0:am1-l1
                    m1 = am1 - l1 - n1
                    index2 = 1
                    for l2 in am2:-1:0
                        for n2 in 0:am2-l2
                            m2 = am2 - l2 - n2
                            out[index1+N1*(index2-1)] +=
                                E[1, 1, l1+1, l2+1] *
                                E[2, 1, m1+1, m2+1] *
                                E[3, 1, n1+1, n2+1] * prefac
                            index2 += 1
                        end
                    end
                    index1 += 1
                end
            end
        end
    end
    return out
end

"""
    batch_S_pair_ref!(out, BS, posA, posB)

Naive reference: for each batch element `b`, compute the full `N × N` overlap
matrix between basis functions centered at `posA[:, b]` and `posB[:, b]`
(positions are plain numeric matrices in Å — this matches the prototype's API,
which predates the package's Unitful-only public API). Allocates per shell pair
block. Correctness oracle for the batched KA kernels.
"""
function batch_S_pair_ref!(out::AbstractArray{T,3}, BS,
                           posA::AbstractMatrix, posB::AbstractMatrix) where {T}
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
    @assert size(out, 1) == Ntot && size(out, 2) == Ntot

    fill!(out, zero(T))
    for b in 1:size(posA, 2)
        𝐫A = SVector{3,Float64}(posA[1, b], posA[2, b], posA[3, b])
        𝐫B = SVector{3,Float64}(posB[1, b], posB[2, b], posB[3, b])
        for s_a in 1:nshells, s_b in 1:nshells
            sa = BS.basis[s_a]
            sb = BS.basis[s_b]
            new_sa = CartesianShell(sa.l, sa.coef, sa.exp,
                                    Molecules.Atom(sa.atom.Z, sa.atom.mass, 𝐫A))
            new_sb = CartesianShell(sb.l, sb.coef, sb.exp,
                                    Molecules.Atom(sb.atom.Z, sb.atom.mass, 𝐫B))
            block = view(out, offset[s_a]+1:offset[s_a]+nbf[s_a],
                              offset[s_b]+1:offset[s_b]+nbf[s_b], b)
            E = alloc_E(new_sa, new_sb)
            generate_S_pair!(block, E, new_sa, new_sb)
        end
    end
    return out
end
