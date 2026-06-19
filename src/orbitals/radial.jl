# Radial basis  R_k(r) = r^{poly[k]} · Σ_m D[k,m] exp(-ζ[k,m] f(r))
#
# A single concrete radial type with a Gaussian (`f(r)=r²`) or Slater (`f(r)=r`)
# decay; an STO is a Slater radial with `K>1` contraction terms. The learnable
# parameters are `ζ` and `D` (both `[nRad × K]`). Evaluation goes through
# KernelAbstractions — the same kernels run on the CPU and GPU backends — and a
# plain forward-only `evaluate_ref` is kept as a testing oracle.

# ---- decay functions ----

abstract type AbstractDecayFunction end
struct GaussianDecay <: AbstractDecayFunction end
struct SlaterDecay   <: AbstractDecayFunction end

(::GaussianDecay)(r) = r^2
(::SlaterDecay)(r)   = r
df(::GaussianDecay, r) = 2r
df(::SlaterDecay, r::T) where {T} = one(T)

const NT_NL = NamedTuple{(:n, :l), Tuple{Int, Int}}

"""
`Rnl` : radial basis with functions
`R_k(r) = r^{poly[k]} · Σ_m D[k,m] exp(-ζ[k,m] f(r))`, where `f` is a
`GaussianDecay` (`f(r)=r²`) or `SlaterDecay` (`f(r)=r`). Learnable parameters are
`ζ` and `D` (size `[nRad × K]`); `K>1` gives a contracted (STO-style) radial.
`natural_indices` are `(n, l)`.
"""
struct Rnl{TM <: AbstractMatrix, DF <: AbstractDecayFunction, LEN} <: AbstractP4MLBasis
    ζ::TM                       # [nRad × K] exponents      (learnable)
    D::TM                       # [nRad × K] coefficients   (learnable)
    poly::Vector{Int}           # polynomial degree per radial function
    decay::DF
    spec::SVector{LEN, NT_NL}   # (n, l) per radial function
end

function Rnl(ζ::AbstractMatrix, D::AbstractMatrix, poly::AbstractVector{<:Integer},
             decay::AbstractDecayFunction, spec::AbstractVector{NT_NL})
    LEN = length(spec)
    @assert size(ζ) == size(D)
    @assert size(ζ, 1) == LEN == length(poly)
    return Rnl{typeof(ζ), typeof(decay), LEN}(
                ζ, D, collect(Int, poly), decay, SVector{LEN, NT_NL}(spec))
end

Base.length(basis::Rnl) = length(basis.spec)
natural_indices(basis::Rnl) = basis.spec
_generate_input(::Rnl) = rand()
Base.show(io::IO, b::Rnl) =
        print(io, "Rnl($(typeof(b.decay).name.name), $(length(b)) fns)")

_valtype(::Rnl, T::Type{<: Number}) = T
_valtype(::Rnl, T::Type{<: Number}, ::Union{Nothing, @NamedTuple{}}, st) = T
_valtype(::Rnl, T::Type{<: Number}, ps, st) =
        promote_type(T, eltype(ps.ζ), eltype(ps.D))

_static_params(basis::Rnl) = (ζ = basis.ζ, D = basis.D)
_init_luxparams(basis::Rnl) = (ζ = Matrix(basis.ζ), D = Matrix(basis.D))

# static (non-trainable) state, analogous to `_static_params`. The fallback is
# empty; sub-bases that carry state (e.g. the SpheriCart `Ylm` `Flm` matrix)
# specialise it (see atomicorbitals.jl).
_static_state(::AbstractP4MLBasis) = NamedTuple()

# ---- shared helpers ----

function _invmap(a::AbstractVector)
    inva = Dict{eltype(a), Int}()
    for i = 1:length(a)
        inva[a[i]] = i
    end
    return inva
end

# copy a host index/structural vector onto the backend of `ref`
_device_like(ref, v::AbstractVector) =
        copyto!(similar(ref, eltype(v), length(v)), v)

# allocate output / gradient arrays on the backend of `X` (KA kernels fill them)
_alloc_val(basis, X, ps, st) =
        similar(X, _valtype(basis, X, ps, st), length(X), length(basis))
_alloc_grad(basis, X, ps, st) =
        similar(X, _gradtype(basis, X, ps, st), length(X), length(basis))

# ---- KA kernels (default path; run on CPU and GPU backends) ----

@kernel function _rnl_val_ka!(R, @Const(r), @Const(ζ), @Const(D), @Const(poly), decay)
    i, k = @index(Global, NTuple)
    K = size(ζ, 2)
    ri = r[i]
    fx = decay(ri)
    s = zero(eltype(R))
    @inbounds for m = 1:K
        s += D[k, m] * exp(-ζ[k, m] * fx)
    end
    @inbounds R[i, k] = ri^poly[k] * s
end

@kernel function _rnl_ed_ka!(R, dR, @Const(r), @Const(ζ), @Const(D), @Const(poly), decay)
    i, k = @index(Global, NTuple)
    K = size(ζ, 2)
    ri = r[i]
    fx = decay(ri)
    dfx = df(decay, ri)
    s = zero(eltype(R))
    ds = zero(eltype(R))
    @inbounds for m = 1:K
        a = D[k, m] * exp(-ζ[k, m] * fx)
        s += a
        ds += -ζ[k, m] * dfx * a
    end
    p = poly[k]
    rp  = ri^p
    drp = (p == 0) ? zero(ri) : p * ri^(p - 1)
    @inbounds R[i, k]  = rp * s
    @inbounds dR[i, k] = drp * s + rp * ds
end

function evaluate(basis::Rnl, r::BATCH, ps, st)
    R = _alloc_val(basis, r, ps, st)
    backend = KA.get_backend(R)
    poly = _device_like(R, basis.poly)
    _rnl_val_ka!(backend)(R, r, ps.ζ, ps.D, poly, basis.decay; ndrange = size(R))
    KA.synchronize(backend)
    return R
end

evaluate(basis::Rnl, r::BATCH) =
        evaluate(basis, r, _static_params(basis), _static_state(basis))

function evaluate_ed(basis::Rnl, r::BATCH, ps, st)
    R  = _alloc_val(basis, r, ps, st)
    dR = _alloc_grad(basis, r, ps, st)
    backend = KA.get_backend(R)
    poly = _device_like(R, basis.poly)
    _rnl_ed_ka!(backend)(R, dR, r, ps.ζ, ps.D, poly, basis.decay; ndrange = size(R))
    KA.synchronize(backend)
    return R, dR
end

evaluate_ed(basis::Rnl, r::BATCH) =
        evaluate_ed(basis, r, _static_params(basis), _static_state(basis))

# ---- plain forward-only reference (testing oracle for the KA forward pass) ----

function evaluate_ref(basis::Rnl, r::AbstractVector, ps = _static_params(basis))
    ζ, D = ps.ζ, ps.D
    nRad, K = size(ζ)
    nX = length(r)
    R = zeros(promote_type(eltype(r), eltype(ζ), eltype(D)), nX, nRad)
    for k = 1:nRad, i = 1:nX
        fx = basis.decay(r[i])
        s = zero(eltype(R))
        for m = 1:K
            s += D[k, m] * exp(-ζ[k, m] * fx)
        end
        R[i, k] = r[i]^basis.poly[k] * s
    end
    return R
end

# ---- parameter pullback (CPU; for Lux/AD training) ----

function pullback_ps(∂R, basis::Rnl, r::BATCH, ps, st)
    ζ, D = ps.ζ, ps.D
    nRad, K = size(ζ)
    nX = length(r)
    ∂ζ = fill!(similar(ζ), 0)
    ∂D = fill!(similar(D), 0)
    for k = 1:nRad, i = 1:nX
        fx = basis.decay(r[i])
        rp = r[i]^basis.poly[k]
        for m = 1:K
            a = exp(-ζ[k, m] * fx)
            ∂D[k, m] += ∂R[i, k] * rp * a
            ∂ζ[k, m] += ∂R[i, k] * rp * D[k, m] * a * (-fx)
        end
    end
    return (ζ = ∂ζ, D = ∂D)
end
