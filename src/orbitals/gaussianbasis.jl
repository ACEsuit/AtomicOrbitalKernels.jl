# Build an `AtomicOrbitals` basis from a (spherical) `GaussianBasis.BasisSet`.
#
# Each chemical ELEMENT in the basis set becomes one species of the resulting
# species-indexed radial: the element's shells fill that species' `(ζ,D)` slice of
# a SHARED `(n,l)` spec — the union over all elements, zero-padded where an element
# lacks a shell. Coefficients are taken verbatim (GaussianBasis already L2-normalizes
# spherical shells, and SpheriCart `SolidHarmonics` are L2-normalized too, so the AO
# values match with no extra constant); `poly = 0` since the `r^l` lives in the solid
# harmonic. Species labels are AtomsBase `ChemicalSpecies`. An element may appear at
# most once (one atom per element) — otherwise this errors.
#
# Orbitals are centre-free: the `BasisSet` atom positions are ignored (only the
# elements and shells are used); evaluate the result at coordinates relative to an
# atom, tagging each input with that atom's species.
function gaussian_orbitals(bs::BasisSet; T = Float64)
    shells = bs.basis
    isempty(shells) && error("the basis set has no shells")
    all(s -> s isa SphericalShell, shells) ||
        error("only spherical GaussianBasis basis sets are supported (use the default \
               `spherical = true`)")
    Zs = [Int(a.Z) for a in bs.atoms]
    allunique(Zs) || error("each chemical element may appear at most once in the \
                            basis set; got atomic numbers $(Zs)")
    NZ = length(Zs)
    zlist = ntuple(i -> ChemicalSpecies(Zs[i]), NZ)
    Lmax = maximum(s.l for s in shells)

    # shared (n,l) spec: per `l`, the maximum shell count over all elements
    nl_count = zeros(Int, Lmax + 1)
    for Z in Zs
        cnt = zeros(Int, Lmax + 1)
        for s in shells
            Int(s.atom.Z) == Z && (cnt[s.l + 1] += 1)
        end
        nl_count .= max.(nl_count, cnt)
    end
    spec = NT_NL[];  nnspec = NT_NNL[]
    for l = 0:Lmax, n = 1:nl_count[l + 1]
        push!(spec,   (n = n, l = l))
        push!(nnspec, (n1 = n, n2 = 1, l = l))
    end
    nRad = length(spec)
    K = maximum(length(s.exp) for s in shells)

    # per-species (ζ,D): fill the slots the element has; padding stays ζ=1, D=0 (inert)
    ζ = ones(T, nRad, K, NZ)
    D = zeros(T, nRad, K, NZ)
    for (σ, Z) in enumerate(Zs)
        ncount = zeros(Int, Lmax + 1)
        for s in shells
            Int(s.atom.Z) == Z || continue
            n = (ncount[s.l + 1] += 1)
            k = findfirst(p -> p.n == n && p.l == s.l, spec)
            nk = length(s.exp)
            @views ζ[k, 1:nk, σ] .= s.exp
            @views D[k, 1:nk, σ] .= s.coef
        end
    end

    radial = GaussianTypeRadials(ζ, D, spec, nnspec, zlist)
    return AtomicOrbitals(radial, _default_ylm(Lmax))
end
