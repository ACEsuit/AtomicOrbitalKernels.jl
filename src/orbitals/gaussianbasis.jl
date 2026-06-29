"""
    gaussian_orbitals(bs::GaussianBasis.BasisSet; length_unit, T = Float64)
    gaussian_orbitals(basisname, elements; length_unit, T = Float64)
    gaussian_orbitals(pairs; length_unit, T = Float64)   # pairs :: [el => name, …]

Build an `AtomicOrbitals` basis from Gaussian basis-set data.

Each chemical **element** becomes one species of the species-indexed radial: its
shells fill that species' `(ζ,D)` slice of a shared `(n,l)` spec — the union over
all elements, zero-padded where an element lacks a shell. Coefficients are taken
verbatim; GaussianBasis L2-normalizes spherical shells and SpheriCart's solid
harmonics are L2-normalized too, so the AO values need no extra constant (`poly =
0`, as the `r^l` lives in the solid harmonic). An element may appear at most once;
only spherical basis sets are supported.

The basis is **centre-free**: atom positions in the basis set are ignored. Evaluate
at coordinates relative to an atom, tagging each input `PState` with that atom's
`ChemicalSpecies`.

# Forms
- `bs::BasisSet` — convert an existing GaussianBasis basis set.
- `basisname, elements` — load `basisname` for every element in `elements`; each
  element is an atomic number, `Symbol`, `String`, or `ChemicalSpecies`.
- `pairs` — a per-element mix of named sets, e.g. `[:C => "cc-pvdz", :H => "sto-3g"]`.

`T` sets the element type of the stored `(ζ,D)` parameters. `length_unit` is
**required** (no default): the unit of the coordinates you will evaluate at —
`:bohr`, `:angstrom`/`:Å`, or a Unitful length. The `(ζ,D)` are stored in atomic
units; positions are scaled into Bohr on input. Forcing an explicit choice avoids
silently assuming a unit.

# Examples
```julia
gaussian_orbitals(BasisSet("cc-pvdz", "C 0 0 0"); length_unit = :bohr)
gaussian_orbitals("cc-pvdz", [:C, :N, :O]; length_unit = :angstrom)
gaussian_orbitals([:C => "cc-pvdz", "H" => "sto-3g"]; length_unit = u"nm")
```
"""
function gaussian_orbitals(bs::BasisSet; length_unit, T = Float64)
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
    return AtomicOrbitals(radial, _default_ylm(Lmax); length_unit = length_unit)
end

# normalize an element label to an AtomsBase `ChemicalSpecies`. AtomsBase already
# constructs one from an atomic number, a `Symbol`, or a `ChemicalSpecies`; a
# string goes through its `Symbol`.
_species(el) = ChemicalSpecies(el)
_species(el::AbstractString) = ChemicalSpecies(Symbol(el))

# one named set loaded for every element; positions are placeholders (centre-free)
function gaussian_orbitals(basisname::AbstractString, elements; length_unit, T = Float64)
    species = [ _species(el) for el in elements ]
    isempty(species) && error("`elements` is empty")
    geom = join(("$(string(sp)) 0.0 0.0 $(2.0 * (i - 1))"
                 for (i, sp) in enumerate(species)), "\n")
    return gaussian_orbitals(BasisSet(basisname, geom); length_unit = length_unit, T = T)
end

# per-element mix: GaussianBasis applies one name per `BasisSet` call, so load each
# element separately and merge the shells.
function gaussian_orbitals(pairs::AbstractVector{<:Pair}; length_unit, T = Float64)
    isempty(pairs) && error("the `element => basis` list is empty")
    subs = [ BasisSet(string(name),
                      "$(string(_species(el))) 0.0 0.0 $(2.0 * (i - 1))")
             for (i, (el, name)) in enumerate(pairs) ]
    atoms = reduce(vcat, (s.atoms for s in subs))
    basis = reduce(vcat, (s.basis for s in subs))
    return gaussian_orbitals(BasisSet("mixed", atoms, basis);
                             length_unit = length_unit, T = T)
end
