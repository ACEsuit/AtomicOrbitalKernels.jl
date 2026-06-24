# Gaussian-type and Slater-type radial bases:
#   R_k(r; σ) = r^{poly[k]} · Σ_m D[k,m,σ] exp(-ζ[k,m,σ] f(r))
# with f(r)=r² (Gaussian) or f(r)=r (Slater); K>1 columns give a contraction.
# `ζ`, `D` are the learnable parameters, carrying a species axis σ: atoms of the
# same species share one `(ζ,D)` slice (weight-sharing). Evaluation runs through
# KernelAbstractions (the same kernels on CPU and GPU); `evaluate_ref` is a
# plain forward-only testing oracle.

const NT_NL  = NamedTuple{(:n, :l), Tuple{Int, Int}}
const NT_NNL = NamedTuple{(:n1, :n2, :l), Tuple{Int, Int, Int}}

abstract type GSRadials <: AbstractP4MLBasis end

struct GaussianTypeRadials{TM <: AbstractArray, LEN, NZ, TZ} <: GSRadials
    ζ::TM                          # [nRad × K × NZ] exponents     (learnable)
    D::TM                          # [nRad × K × NZ] coefficients  (learnable)
    spec::SVector{LEN, NT_NL}      # (n, l) per radial function
    nnspec::SVector{LEN, NT_NNL}   # (n1, n2, l) provenance per radial function
    zlist::NTuple{NZ, TZ}          # species labels; species axis σ ↔ zlist[σ]
end

struct SlaterTypeRadials{TM <: AbstractArray, LEN, NZ, TZ} <: GSRadials
    ζ::TM
    D::TM
    spec::SVector{LEN, NT_NL}
    nnspec::SVector{LEN, NT_NNL}
    zlist::NTuple{NZ, TZ}
end

for TR in (:GaussianTypeRadials, :SlaterTypeRadials)
    @eval function $TR(ζ, D, spec, nnspec, zlist)
        LEN = length(spec)
        nRad, K, NZ = size(ζ)
        @assert size(ζ) == size(D)
        @assert nRad == LEN == length(nnspec)
        @assert NZ == length(zlist)
        # store ζ,D as a statically-sized `MArray` (the sizes are recorded in the
        # basis type — a latent enabler for size-specialised kernels); the params
        # are converted back to a plain `Array` for `ps` (see `_static_params`).
        ζm = MArray{Tuple{nRad, K, NZ}}(ζ)
        Dm = MArray{Tuple{nRad, K, NZ}}(D)
        zt = Tuple(zlist)
        return $TR{typeof(ζm), LEN, NZ, eltype(zt)}(ζm, Dm,
                    SVector{LEN, NT_NL}(spec), SVector{LEN, NT_NNL}(nnspec), zt)
    end
end

# --- species / DecoratedParticles input helpers ---
# A `PState` carries position `x.𝐫` (an `SVector{3}`) and species `x.S`. The
# internal kernel layers consume already-extracted batches (radii `r` or
# positions `Rs`, plus species indices `sidx`); the public layers take a vector
# `X` of either coordinates (→ species 1) or `PState`s (→ species from `x.S`).
_position(x::PState) = x.𝐫
_radii(X::AbstractVector{<:PState}) = (norm ∘ _position).(X)
_radii(r::AbstractVector{<:Real}) = r

# species label → species-axis index σ ∈ 1:NZ (host side)
function _z2i(basis::GSRadials, s)
    σ = findfirst(==(s), basis.zlist)
    σ === nothing && error("species $(s) not in basis species list $(basis.zlist)")
    return σ
end

# species indices for an input batch: from the `PState` species, else default to
# species 1 (plain coordinates carry no species).
_species_indices(basis::GSRadials, X::AbstractVector{<:PState}) =
        [ _z2i(basis, x.S) for x in X ]

# any non-`PState` input (radii for the radial layer, positions when the orbital
# delegates here) carries no species → default to species 1, on the input backend.
_species_indices(basis::GSRadials, X::AbstractVector) =
        fill!(similar(X, Int, length(X)), 1)



# decay form, dispatched by family. The kernel is passed a `Val` tag (isbits, so
# it works on the GPU); `_decay(basis, r)` is the host-side convenience.
_decaytag(::GaussianTypeRadials) = Val(:gaussian)
_decaytag(::SlaterTypeRadials)   = Val(:slater)
@inline _decay(::Val{:gaussian}, r) = r^2
@inline _decay(::Val{:slater},   r) = r
@inline _ddecay(::Val{:gaussian}, r) = 2r
@inline _ddecay(::Val{:slater},  r::T) where {T} = one(T)
@inline _decay(b::GSRadials, r)  = _decay(_decaytag(b), r)
@inline _ddecay(b::GSRadials, r) = _ddecay(_decaytag(b), r)

Base.length(basis::GSRadials) = length(basis.spec)
nspecies(basis::GSRadials) = length(basis.zlist)
natural_indices(basis::GSRadials) = basis.spec
_generate_input(::GSRadials) = rand()
Base.show(io::IO, b::GSRadials) = print(io, "$(nameof(typeof(b)))($(length(b)) fns)")

_valtype(::GSRadials, T::Type{<: Number}) = T
_valtype(::GSRadials, T::Type{<: Number}, ::Union{Nothing, @NamedTuple{}}, st) = T
_valtype(::GSRadials, T::Type{<: Number}, ps, st) =
        promote_type(T, eltype(ps.ζ), eltype(ps.D))

# params are extracted as plain `Array`s: an `MArray` can't move to the GPU and
# the kernels want dynamic-size, AD-friendly arrays — the static MArray sizes
# live only in the basis type.
_static_params(b::GSRadials) = (ζ = Array(b.ζ), D = Array(b.D))
_init_luxparams(b::GSRadials) = (ζ = Array(b.ζ), D = Array(b.D))
# the radial power `r^poly[k]` is derived, not stored: with solid harmonics the
# `r^l` is already inside `Z_lm`, so a GTO has no radial power (poly = 0) while an
# STO keeps `r^(n-1-l)`; here `nnspec.n1 = n - l`, hence `poly = n1 - 1`.
_powers(b::GaussianTypeRadials{TM, LEN}) where {TM, LEN} = zero(SVector{LEN, Int})
_powers(b::SlaterTypeRadials{TM, LEN}) where {TM, LEN} =
        SVector{LEN, Int}(nl.n1 - 1 for nl in b.nnspec)

# the polynomial degrees are non-trainable state (so they move to the device
# with the rest of the state rather than being copied on every call).
_static_state(b::GSRadials) = (poly = _powers(b),)
_init_luxstate(b::GSRadials) = _static_state(b)

# ---- KA kernels (default path; CPU and GPU backends) ----

@kernel function _rnl_val_ka!(R, @Const(r), @Const(sidx), @Const(ζ), @Const(D),
                              @Const(poly), tag)
    i, k = @index(Global, NTuple)
    K = size(ζ, 2)
    σ = sidx[i]
    ri = r[i]
    fx = _decay(tag, ri)
    s = zero(eltype(R))
    # TODO: can this be unrolled? Is it worth being unrolled? 
    @inbounds for m = 1:K
        s += D[k, m, σ] * exp(-ζ[k, m, σ] * fx)
    end
    @inbounds R[i, k] = ri^poly[k] * s
end

@kernel function _rnl_ed_ka!(R, dR, @Const(r), @Const(sidx), @Const(ζ), @Const(D),
                             @Const(poly), tag)
    i, k = @index(Global, NTuple)
    K = size(ζ, 2)
    σ = sidx[i]
    ri = r[i]
    fx = _decay(tag, ri)
    dfx = _ddecay(tag, ri)
    s = zero(eltype(R))
    ds = zero(eltype(R))
    # TODO: can this be unrolled?
    @inbounds for m = 1:K
        a = D[k, m, σ] * exp(-ζ[k, m, σ] * fx)
        s += a
        ds += -ζ[k, m, σ] * dfx * a
    end
    p = poly[k]
    rp  = ri^p
    drp = (p == 0) ? zero(ri) : p * ri^(p - 1)
    @inbounds R[i, k]  = rp * s
    @inbounds dR[i, k] = drp * s + rp * ds
end

# Two input layers: an internal kernel-facing layer taking radii `r` + species
# indices `sidx` (both already on the target backend), and a public layer taking
# a vector `X` of radii (→ species 1) or `PState`s (→ species from `x.S`), which
# extracts `r`/`sidx` on the host and delegates.

function evaluate(basis::GSRadials, r::AbstractVector{<:Real},
                  sidx::AbstractVector{<:Integer}, ps, st)
    R = _alloc_val(basis, r, ps, st)
    backend = KA.get_backend(R)
    _rnl_val_ka!(backend)(R, r, sidx, ps.ζ, ps.D, st.poly, _decaytag(basis);
                          ndrange = size(R))
    KA.synchronize(backend)
    return R
end

evaluate(basis::GSRadials, X::AbstractVector, ps, st) =
        evaluate(basis, _radii(X), _species_indices(basis, X), ps, st)

evaluate(basis::GSRadials, X::AbstractVector) =
        evaluate(basis, X, _static_params(basis), _static_state(basis))

function evaluate_ed(basis::GSRadials, r::AbstractVector{<:Real},
                     sidx::AbstractVector{<:Integer}, ps, st)
    R  = _alloc_val(basis, r, ps, st)
    dR = _alloc_grad(basis, r, ps, st)
    backend = KA.get_backend(R)
    _rnl_ed_ka!(backend)(R, dR, r, sidx, ps.ζ, ps.D, st.poly, _decaytag(basis);
                         ndrange = size(R))
    KA.synchronize(backend)
    return R, dR
end

evaluate_ed(basis::GSRadials, X::AbstractVector, ps, st) =
        evaluate_ed(basis, _radii(X), _species_indices(basis, X), ps, st)

evaluate_ed(basis::GSRadials, X::AbstractVector) =
        evaluate_ed(basis, X, _static_params(basis), _static_state(basis))

# ---- plain forward-only reference (testing oracle) ----

function evaluate_ref(basis::GSRadials, r::AbstractVector{<:Real},
                      sidx::AbstractVector{<:Integer},
                      ps = _static_params(basis), st = _static_state(basis))
    ζ, D = ps.ζ, ps.D
    poly = st.poly
    nRad, K, NZ = size(ζ)
    nX = length(r)
    R = zeros(promote_type(eltype(r), eltype(ζ), eltype(D)), nX, nRad)
    for k = 1:nRad, i = 1:nX
        σ = sidx[i]
        fx = _decay(basis, r[i])
        s = zero(eltype(R))
        for m = 1:K
            s += D[k, m, σ] * exp(-ζ[k, m, σ] * fx)
        end
        R[i, k] = r[i]^poly[k] * s
    end
    return R
end

evaluate_ref(basis::GSRadials, X::AbstractVector,
             ps = _static_params(basis), st = _static_state(basis)) =
        evaluate_ref(basis, _radii(X), _species_indices(basis, X), ps, st)

# ---- parameter pullback ----

# parameter pullback. ∂ζ[k,m,σ] / ∂D[k,m,σ] are reductions over the points of
# species σ; a per-point atomic scatter has high contention per slot. We keep the
# *segmented* reduction (low contention) and add the species axis: ndrange is
# (NZ, ng, nRad, K), where each (σ,gi,k,m) work-item strides over the points,
# accumulates only those with sidx==σ in registers, and contributes a single
# atomic per slot — contention is ng-way per species, not nX-way. ng ≈ 128
# (capped at nX). The single-species segmented kernel measured ~6–30× faster than
# per-point atomics on an A100; cost of the species axis is the NZ factor + a
# branch (benchmarked against a plain (i,k,m) atomic scatter — see benchmark/).
const _PB_NGROUPS = 128

@kernel function _gtostoradials_pb_ka!(∂ζ, ∂D, @Const(∂R), @Const(r), @Const(sidx),
                              @Const(ζ), @Const(D), @Const(poly), tag, ng)
    σ, gi, k, m = @index(Global, NTuple)
    nX = length(r)
    @inbounds begin
        ζkm = ζ[k, m, σ]; Dkm = D[k, m, σ]
        pζ = zero(eltype(∂ζ)); pD = zero(eltype(∂D))
        i = gi
        while i <= nX
            if sidx[i] == σ
                ri = r[i]; fx = _decay(tag, ri)
                a  = exp(-ζkm * fx)
                gg = ∂R[i, k] * ri^poly[k]
                pD += gg * a
                pζ += gg * Dkm * a * (-fx)
            end
            i += ng
        end
        KA.@atomic ∂D[k, m, σ] += pD
        KA.@atomic ∂ζ[k, m, σ] += pζ
    end
end

function pullback_ps(∂R, basis::GSRadials, r::AbstractVector{<:Real},
                     sidx::AbstractVector{<:Integer}, ps, st)
    T = promote_type(eltype(∂R), eltype(ps.ζ), eltype(ps.D))
    ∂ζ = fill!(similar(ps.ζ, T), zero(T))
    ∂D = fill!(similar(ps.D, T), zero(T))
    backend = KA.get_backend(∂ζ)
    nRad, K, NZ = size(ps.ζ)
    ng = min(_PB_NGROUPS, length(r))
    _gtostoradials_pb_ka!(backend)(∂ζ, ∂D, ∂R, r, sidx, ps.ζ, ps.D, st.poly,
                            _decaytag(basis), ng; ndrange = (NZ, ng, nRad, K))
    KA.synchronize(backend)
    return (ζ = ∂ζ, D = ∂D)
end

pullback_ps(∂R, basis::GSRadials, X::AbstractVector, ps, st) =
        pullback_ps(∂R, basis, _radii(X), _species_indices(basis, X), ps, st)
