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

# normalize an element label to an AtomsBase `ChemicalSpecies`. AtomsBase already
# constructs one from an atomic number, a `Symbol`, or a `ChemicalSpecies`; a
# string goes through its `Symbol`.
_species(el) = ChemicalSpecies(el)
_species(el::AbstractString) = ChemicalSpecies(Symbol(el))

# Convenience: build the basis from a basis-set name and a list of `elements`
# (each an atomic number, symbol, string, or `ChemicalSpecies`); the same named
# set is loaded for every element via GaussianBasis. Orbitals are centre-free, so
# the (well-separated) placeholder positions are irrelevant.
function gaussian_orbitals(basisname::AbstractString, elements; T = Float64)
    species = [ _species(el) for el in elements ]
    isempty(species) && error("`elements` is empty")
    geom = join(("$(string(sp)) 0.0 0.0 $(2.0 * (i - 1))"
                 for (i, sp) in enumerate(species)), "\n")
    return gaussian_orbitals(BasisSet(basisname, geom); T = T)
end

# Convenience: a per-element mix of named basis sets, given as `element => name`
# pairs (e.g. `[:C => "cc-pvdz", :H => "sto-3g"]`). GaussianBasis applies one name
# per call, so we load each element separately and merge the shells.
function gaussian_orbitals(pairs::AbstractVector{<:Pair}; T = Float64)
    isempty(pairs) && error("the `element => basis` list is empty")
    subs = [ BasisSet(string(name),
                      "$(string(_species(el))) 0.0 0.0 $(2.0 * (i - 1))")
             for (i, (el, name)) in enumerate(pairs) ]
    atoms = reduce(vcat, (s.atoms for s in subs))
    basis = reduce(vcat, (s.basis for s in subs))
    return gaussian_orbitals(BasisSet("mixed", atoms, basis); T = T)
end
