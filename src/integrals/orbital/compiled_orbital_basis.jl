# Species-aware Cartesian compiled basis built from an `AtomicOrbitals` basis.
#
# The orbital side is *per-species*: a shared `(n,l)` radial spec with `(ζ,D)`
# carrying a species axis σ (`[nRad × K × NZ]`). This struct mirrors that layout
# for the Cartesian overlap kernels — one Cartesian shell per `(n,l)` radial,
# shared shell structure across species, `(ζ,D)` kept per species so the learned
# parameters are recoverable (transfer back to the `AtomicOrbitals` basis).
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
    CompiledOrbitalBasis{T,VI,A3,NZ,TZ}

Species-aware, Cartesian struct-of-arrays form of a Gaussian `AtomicOrbitals`
basis, produced by [`compile_basis`](@ref). One Cartesian shell per `(n,l)`
radial; the shell structure (`ls`, `nbf`, offsets) is shared across species and
`(ζ,D)` carry the species axis. Not exported.

Fields:

- `Lmax`         : maximum angular momentum
- `nshells`      : number of shells (= number of radials)
- `K`            : primitive slots per shell (padded; unused slots have `D=0`)
- `ls`           : `l` per shell, length `nshells`
- `nbf`          : Cartesian functions per shell `(l+1)(l+2)÷2`, length `nshells`
- `basis_offset` : prefix-sum of `nbf`, length `nshells+1`
- `nbf_total`    : total Cartesian functions (per species)
- `ζ`            : exponents `[nshells × K × NZ]` (kernel `α`; also a learnable param)
- `D`            : spherical coefficients `[nshells × K × NZ]` (learnable param, kept recoverable)
- `coef`         : Cartesian-normalized coefficients `[nshells × K × NZ]` (kernel-facing, derived from `ζ,D`)
- `zlist`        : species labels; species axis σ ↔ `zlist[σ]`
"""
struct CompiledOrbitalBasis{T, VI<:AbstractVector{Int}, A3<:AbstractArray{T,3},
                            NZ, TZ}
    Lmax::Int
    nshells::Int
    K::Int
    ls::VI
    nbf::VI
    basis_offset::VI
    nbf_total::Int
    ζ::A3
    D::A3
    coef::A3
    zlist::NTuple{NZ, TZ}
end

nspecies(b::CompiledOrbitalBasis) = length(b.zlist)
Base.show(io::IO, b::CompiledOrbitalBasis) =
        print(io, "CompiledOrbitalBasis($(b.nshells) shells, $(b.nbf_total) bfs, ",
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
    compile_basis(orb::AtomicOrbitals, ::Type{T}=Float64) -> CompiledOrbitalBasis

Compile a Gaussian `AtomicOrbitals` basis into a species-aware Cartesian
[`CompiledOrbitalBasis`](@ref) for the batched overlap kernels. Each `(n,l)`
radial becomes a Cartesian shell of momentum `l`; the spherical coefficients `D`
are renormalized to GaussianBasis's Cartesian convention so the overlaps match
`BasisSet(...; spherical=false)`. `(ζ,D)` are stored verbatim so the parameters
transfer back to the orbital basis.

Only Gaussian-type orbital bases are supported (Slater radials carry a radial
power the overlap kernels do not integrate).
"""
function compile_basis(orb::GaussianTypeOrbitals, ::Type{T}=Float64) where {T}
    rad  = orb.Rnl
    spec = rad.spec
    ζin  = Array{T}(rad.ζ)
    Din  = Array{T}(rad.D)
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

    return CompiledOrbitalBasis{T, Vector{Int}, Array{T,3}, NZ, eltype(rad.zlist)}(
        Lmax, nshells, K, ls, nbf, basis_offset, basis_offset[end],
        ζin, Din, coef, rad.zlist)
end

compile_basis(::SlaterTypeOrbitals, ::Type{T}=Float64) where {T} =
        error("the overlap kernels integrate plain Gaussians; a Slater-type \
               orbital basis carries a nonzero radial power and is unsupported")
