# Species-aware Cartesian-Gaussian compiled basis built from an `AtomicOrbitals`
# basis (Cartesian Gaussian-type orbitals).
#
# The orbital side is *per-species*: a shared `(n,l)` radial spec with `(ζ,D)`
# carrying a species axis σ (`[nRad × K × NZ]`). This struct mirrors that layout
# for the Cartesian overlap kernels — one Cartesian shell per `(n,l)` radial,
# shared shell structure across species. It is a *derived* artifact: it stores
# only the kernel inputs `ζ` (exponents) and `coef` (Cartesian-normalized
# coefficients) per species. The learnable parameters `(ζ,D)` live in the source
# `AtomicOrbitals`; overlap gradients flow back through `compile_basis` (the
# differentiable spherical→Cartesian map), so `D` is not stored here.
#
# Gaussian radials only: the overlap kernels integrate plain Gaussians, while a
# Slater radial carries a nonzero radial power `r^(n1-1)`.
#
# Normalization: `gaussian_orbitals` stores `D` = GaussianBasis *spherical* coef
# (verbatim). The overlap kernels want GaussianBasis *Cartesian* shells, which
# differ only by a per-shell contraction normalization (the exponents are
# identical). `normalize_cartesian!`'s contraction step applied to `D` reproduces
# GaussianBasis's Cartesian coef exactly (the per-primitive factor cancels):
#   coef = D / sqrt( (df/2^l) Σ_ij D_i D_j / (ζ_i+ζ_j)^(l+3/2) ),  df = π^{3/2}(2l-1)!!
# so `compile_basis(::AtomicOrbitals)` overlaps match `BasisSet(...; spherical=false)`.

"""
    CartesianGTOBasis{T,VI,A3,NZ,TZ}

Species-aware, Cartesian struct-of-arrays form of a Gaussian `AtomicOrbitals`
basis (Cartesian Gaussian-type orbitals), produced by [`compile_basis`](@ref).
One Cartesian shell per `(n,l)` radial; the shell structure (`ls`, `nbf`,
offsets) is shared across species and `ζ`/`coef` carry the species axis. A
derived artifact — the learnable `(ζ,D)` live in the source `AtomicOrbitals`. Not
exported.

Fields:

- `Lmax`         : maximum angular momentum
- `nshells`      : number of shells (= number of radials)
- `K`            : primitive slots per shell (padded; unused slots have `coef=0`)
- `ls`           : `l` per shell, length `nshells`
- `nbf`          : Cartesian functions per shell `(l+1)(l+2)÷2`, length `nshells`
- `basis_offset` : prefix-sum of `nbf`, length `nshells+1`
- `nbf_total`    : total Cartesian functions (per species)
- `ζ`            : Cartesian exponents `[nshells × K × NZ]` (the kernel `α`; identical to the orbital `ζ`)
- `coef`         : Cartesian-normalized coefficients `[nshells × K × NZ]` (kernel-facing, derived from `ζ,D`)
- `zlist`        : species labels; species axis σ ↔ `zlist[σ]`
- `lengthscale`  : factor mapping input positions → native Bohr (from the orbital basis)
"""
struct CartesianGTOBasis{T, VI<:AbstractVector{Int}, A3<:AbstractArray{T,3},
                         NZ, TZ}
    Lmax::Int
    nshells::Int
    K::Int
    ls::VI
    nbf::VI
    basis_offset::VI
    nbf_total::Int
    ζ::A3
    coef::A3
    zlist::NTuple{NZ, TZ}
    lengthscale::T          # input length unit → native (Bohr); from the orbital basis
end

nspecies(b::CartesianGTOBasis) = length(b.zlist)
Base.show(io::IO, b::CartesianGTOBasis) =
        print(io, "CartesianGTOBasis($(b.nshells) shells, $(b.nbf_total) bfs, ",
                  "$(nspecies(b)) species)")

# (2l-1)!! for odd argument (1 for l=0).
_cart_dfac(n::Integer) = (p = 1.0; k = n; while k > 1; p *= k; k -= 2; end; p)

# GaussianBasis Cartesian contraction self-overlap of a single shell's primitives
# `(D, ζ)` at angular momentum `l`. Padded slots (`D=0`) contribute nothing.
# Computed in the working float type (AD-friendly; no hard-coded `Float64`).
function _cart_normsq(D::AbstractVector, ζ::AbstractVector, l::Integer)
    R = float(promote_type(eltype(D), eltype(ζ)))
    df = R(π)^R(3//2) * (l == 0 ? one(R) : R(_cart_dfac(2l - 1)))
    e  = R(l) + R(3//2)
    s  = zero(R)
    @inbounds for i in eachindex(D), j in eachindex(D)
        s += R(D[i]) * R(D[j]) / (R(ζ[i]) + R(ζ[j]))^e
    end
    return (df / R(2)^l) * s
end

"""
    compile_basis(orb::AtomicOrbitals) -> CartesianGTOBasis

Compile a Gaussian `AtomicOrbitals` basis into a species-aware Cartesian
[`CartesianGTOBasis`](@ref) for the batched overlap kernels. Each `(n,l)` radial
becomes a Cartesian shell of momentum `l`; the spherical coefficients `D` are
renormalized to GaussianBasis's Cartesian convention so the overlaps match
`BasisSet(...; spherical=false)`. The element type is inferred from `(ζ,D)`.

The learnable parameters `(ζ,D)` stay in the `AtomicOrbitals`; this map is
differentiable, so overlap gradients flow back through it to `(ζ,D)` and the
compiled basis need not store `D`.

Only Gaussian-type orbital bases are supported (Slater radials carry a radial
power the overlap kernels do not integrate).
"""
function compile_basis(orb::GaussianTypeOrbitals)
    rad  = orb.Rnl
    spec = rad.spec
    ζin  = Array(rad.ζ)
    Din  = Array(rad.D)
    T    = promote_type(eltype(ζin), eltype(Din))
    nshells, K, NZ = size(ζin)
    @assert nshells == length(spec)

    ls   = collect(Int, (s.l for s in spec))
    Lmax = maximum(ls)
    nbf  = Int[ (l + 1) * (l + 2) ÷ 2 for l in ls ]
    basis_offset = Vector{Int}(undef, nshells + 1)
    basis_offset[1] = 0
    for i in 1:nshells
        basis_offset[i+1] = basis_offset[i] + nbf[i]
    end

    coef = zeros(T, nshells, K, NZ)
    for σ in 1:NZ, k in 1:nshells
        Dk = @view Din[k, :, σ]
        ns = _cart_normsq(Dk, (@view ζin[k, :, σ]), ls[k])
        ns > 0 && (@views coef[k, :, σ] .= Dk ./ sqrt(T(ns)))
    end

    return CartesianGTOBasis{T, Vector{Int}, Array{T,3}, NZ, eltype(rad.zlist)}(
        Lmax, nshells, K, ls, nbf, basis_offset, basis_offset[end],
        ζin, coef, rad.zlist, T(orb.lengthscale))
end

compile_basis(::SlaterTypeOrbitals) =
        error("the overlap kernels integrate plain Gaussians; a Slater-type \
               orbital basis carries a nonzero radial power and is unsupported")
