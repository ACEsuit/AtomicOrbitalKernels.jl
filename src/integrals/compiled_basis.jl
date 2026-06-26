# Compiled, type-stable, GPU-shippable form of a `GaussianBasis.BasisSet`.
# Heterogeneous shells are flattened into struct-of-arrays form with prefix-sum
# offsets, so all fields are concrete-eltype arrays — trivially adaptable to a
# device array constructor via `adapt_basis`.
#
# Convention: `nbf[i]` is the number of Cartesian basis functions per shell
# `(l+1)(l+2)÷2` (matching `GaussianBasis.num_basis` for Cartesian shells). The
# kernel contraction uses a linear stride of `2l+1` to match the existing naive
# reference (`Reference.generate_S_pair!`) exactly — for L ≤ 1 these agree, for
# L ≥ 2 they differ and the aliasing is preserved on purpose.
#
# Note: at runtime the kernels are GaussianBasis-free; only `compile_basis(BS)`
# touches the GaussianBasis types. A future extension could move that boundary
# into a package extension to keep inference-only deployments lighter.

"""
    CompiledBasis{T,VI,VT}

Type-stable, struct-of-arrays form of a basis set produced by
[`compile_basis`](@ref). Not exported — users obtain instances by calling
`compile_basis`. Adaptable to any backend array type via [`adapt_basis`](@ref).

Fields:

- `Lmax`           : maximum angular momentum across the basis
- `nshells`        : number of shells
- `ls`             : `l` value per shell, length `nshells`
- `nprim`          : number of primitives per shell, length `nshells`
- `prim_offset`    : prefix-sum into `coef`/`α`, length `nshells+1`
- `coef`           : flat contraction coefficients, length `sum(nprim)`
- `α`              : flat Gaussian exponents (named `α` to avoid shadowing `Base.exp`)
- `nbf`            : Cartesian basis functions per shell, length `nshells`
- `basis_offset`   : prefix-sum of `nbf`, length `nshells+1`
- `nbf_total`      : total number of Cartesian basis functions
"""
struct CompiledBasis{T,VI<:AbstractVector{Int},VT<:AbstractVector{T}}
    Lmax::Int
    nshells::Int
    ls::VI
    nprim::VI
    prim_offset::VI
    coef::VT
    α::VT
    nbf::VI
    basis_offset::VI
    nbf_total::Int
end

"""
    compile_basis(BS::GaussianBasis.BasisSet, ::Type{T}=Float64) -> CompiledBasis

Walk `BS.basis` once (absorbing the type-instability outside the hot loop) and
return a [`CompiledBasis`](@ref) with all fields as concrete-eltype `Vector`s,
ready to be passed to [`batch_overlap!`](@ref) / [`batch_overlap_3c!`](@ref) or
moved to a device via [`adapt_basis`](@ref).

The basis must use Cartesian shells (`spherical=false` in `BasisSet`); the
batched kernels only support Cartesian basis output for now.
"""
function compile_basis(BS, ::Type{T}=Float64) where {T}
    nshells = length(BS.basis)
    ls    = Vector{Int}(undef, nshells)
    nprim = Vector{Int}(undef, nshells)
    nbf   = Vector{Int}(undef, nshells)
    Lmax = 0
    nprim_total = 0
    for i in 1:nshells
        s = BS.basis[i]
        ls[i]    = s.l
        nprim[i] = length(s.coef)
        nbf[i]   = (s.l + 1) * (s.l + 2) ÷ 2
        Lmax = max(Lmax, s.l)
        nprim_total += nprim[i]
    end
    prim_offset = Vector{Int}(undef, nshells + 1)
    prim_offset[1] = 0
    for i in 1:nshells
        prim_offset[i+1] = prim_offset[i] + nprim[i]
    end
    coef = Vector{T}(undef, nprim_total)
    α    = Vector{T}(undef, nprim_total)
    for i in 1:nshells
        s = BS.basis[i]
        off = prim_offset[i]
        for k in 1:nprim[i]
            coef[off + k] = s.coef[k]
            α[off + k]    = s.exp[k]
        end
    end
    basis_offset = Vector{Int}(undef, nshells + 1)
    basis_offset[1] = 0
    for i in 1:nshells
        basis_offset[i+1] = basis_offset[i] + nbf[i]
    end
    return CompiledBasis{T,Vector{Int},Vector{T}}(
        Lmax, nshells, ls, nprim, prim_offset, coef, α,
        nbf, basis_offset, basis_offset[end]
    )
end
