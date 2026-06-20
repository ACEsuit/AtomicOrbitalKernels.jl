# Reading Gaussian basis-set parameters from Basis Set Exchange (BSE) JSON.
#
# Bundled, redistributable (CC-BY) JSON files live in `data/basis/<name>.json`
# (see `data/basis/README.md`). This is the eval-side path: parsed contracted
# shells → learnable `GaussianTypeOrbitals`. On-demand download + local caching
# of arbitrary basis sets is a planned follow-up (see DataPlan.md).

const _BASIS_DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "basis"))

# one contracted Gaussian shell of a single angular momentum `l`
struct ParsedShell{T}
    l::Int
    exponents::Vector{T}        # K primitive exponents
    coefficients::Vector{T}     # K contraction coefficients (for this l)
end

# the contracted-Gaussian shells for one element `Z` of a named basis set
struct ParsedBasis{T}
    name::String
    Z::Int
    shells::Vector{ParsedShell{T}}
end

Base.show(io::IO, pb::ParsedBasis) =
        print(io, "ParsedBasis($(pb.name), Z=$(pb.Z), $(length(pb.shells)) shells)")

"""
    bundled_basis_names() -> Vector{String}

Names of the basis sets bundled with the package (readable offline).
"""
bundled_basis_names() =
        sort([splitext(f)[1] for f in readdir(_BASIS_DATA_DIR) if endswith(f, ".json")])

_basis_file(name::AbstractString) = joinpath(_BASIS_DATA_DIR, lowercase(name) * ".json")

"""
    load_basis(name, Z; T = Float64) -> ParsedBasis

Read the contracted-Gaussian shells for element `Z` of the bundled basis set
`name` (e.g. `"sto-3g"`). A shell listing several angular momenta (e.g. an `sp`
shell) is split into one `ParsedShell` per `l`, sharing the exponents.
"""
function load_basis(name::AbstractString, Z::Integer; T::Type = Float64)
    file = _basis_file(name)
    isfile(file) || error("basis '$name' is not bundled; have: " *
                          join(bundled_basis_names(), ", "))
    data = JSON3.read(read(file, String))
    ekey = Symbol(string(Int(Z)))
    haskey(data.elements, ekey) ||
        error("element Z=$Z not present in bundled basis '$name'")
    shells = ParsedShell{T}[]
    for sh in data.elements[ekey].electron_shells
        exps = parse.(T, sh.exponents)
        for (j, l) in enumerate(sh.angular_momentum)
            push!(shells, ParsedShell{T}(Int(l), exps, parse.(T, sh.coefficients[j])))
        end
    end
    return ParsedBasis{T}(String(data.name), Int(Z), shells)
end

"""
    gaussian_orbitals(pb::ParsedBasis) -> GaussianTypeOrbitals
    gaussian_orbitals(name::AbstractString, Z::Integer; T = Float64)

Build a learnable Gaussian-type orbital basis from parsed shells. Each shell
becomes one radial `R(r) = Σ_m D[m] exp(-ζ[m] r²)`; the `r^l` factor lives in the
SpheriCart solid harmonics, so `poly = 0`. Shells are padded to a common
contraction length `K` with zero coefficients (inert). Coefficients are taken as
published in the basis set — primitive/overall normalisation is **not** applied,
and the angular normalisation is SpheriCart's.
"""
function gaussian_orbitals(pb::ParsedBasis{T}) where {T}
    isempty(pb.shells) && error("basis '$(pb.name)' has no shells for Z=$(pb.Z)")
    nRad = length(pb.shells)
    K = maximum(length(s.exponents) for s in pb.shells)
    ζ = zeros(T, nRad, K)
    D = zeros(T, nRad, K)
    poly = zeros(Int, nRad)
    spec = NT_NL[]
    nnspec = NT_NNL[]
    ncount = Dict{Int, Int}()      # running index per l → unique (n, l)
    for (k, s) in enumerate(pb.shells)
        nk = length(s.exponents)
        @views ζ[k, 1:nk] .= s.exponents
        @views ζ[k, (nk + 1):K] .= one(T)      # padding (D = 0 there, so inert)
        @views D[k, 1:nk] .= s.coefficients
        n = (ncount[s.l] = get(ncount, s.l, 0) + 1)
        push!(spec,   (n = n, l = s.l))
        push!(nnspec, (n1 = n, n2 = 1, l = s.l))
    end
    Lmax = maximum(s.l for s in pb.shells)
    radial = GaussianTypeRadials(ζ, D, poly, spec, nnspec)
    return AtomicOrbitals(radial, _default_ylm(Lmax))
end

gaussian_orbitals(name::AbstractString, Z::Integer; T::Type = Float64) =
        gaussian_orbitals(load_basis(name, Z; T = T))
