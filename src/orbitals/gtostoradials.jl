# Gaussian-type and Slater-type radial bases:
#   R_k(r) = r^{poly[k]} · Σ_m D[k,m] exp(-ζ[k,m] f(r))
# with f(r)=r² (Gaussian) or f(r)=r (Slater); K>1 columns give a contraction.
# `ζ`, `D` are the learnable parameters. Evaluation runs through
# KernelAbstractions (the same kernels on CPU and GPU); `evaluate_ref` is a
# plain forward-only testing oracle.

const NT_NL  = NamedTuple{(:n, :l), Tuple{Int, Int}}
const NT_NNL = NamedTuple{(:n1, :n2, :l), Tuple{Int, Int, Int}}

abstract type GSRadials <: AbstractP4MLBasis end

struct GaussianTypeRadials{TM <: AbstractMatrix, LEN} <: GSRadials
    ζ::TM                          # [nRad × K] exponents      (learnable)
    D::TM                          # [nRad × K] coefficients   (learnable)
    poly::Vector{Int}              # polynomial degree per radial function
    spec::SVector{LEN, NT_NL}      # (n, l) per radial function
    nnspec::SVector{LEN, NT_NNL}   # (n1, n2, l) provenance per radial function
end

struct SlaterTypeRadials{TM <: AbstractMatrix, LEN} <: GSRadials
    ζ::TM
    D::TM
    poly::Vector{Int}
    spec::SVector{LEN, NT_NL}
    nnspec::SVector{LEN, NT_NNL}
end

for TR in (:GaussianTypeRadials, :SlaterTypeRadials)
    @eval function $TR(ζ, D, poly, spec, nnspec)
        LEN = length(spec)
        @assert size(ζ) == size(D)
        @assert size(ζ, 1) == LEN == length(poly) == length(nnspec)
        return $TR{typeof(ζ), LEN}(ζ, D, collect(Int, poly),
                    SVector{LEN, NT_NL}(spec), SVector{LEN, NT_NNL}(nnspec))
    end
end

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
natural_indices(basis::GSRadials) = basis.spec
_generate_input(::GSRadials) = rand()
Base.show(io::IO, b::GSRadials) = print(io, "$(nameof(typeof(b)))($(length(b)) fns)")

_valtype(::GSRadials, T::Type{<: Number}) = T
_valtype(::GSRadials, T::Type{<: Number}, ::Union{Nothing, @NamedTuple{}}, st) = T
_valtype(::GSRadials, T::Type{<: Number}, ps, st) =
        promote_type(T, eltype(ps.ζ), eltype(ps.D))

_static_params(b::GSRadials) = (ζ = b.ζ, D = b.D)
_init_luxparams(b::GSRadials) = (ζ = Matrix(b.ζ), D = Matrix(b.D))
# the polynomial degrees are non-trainable state (so they move to the device
# with the rest of the state rather than being copied on every call).
_static_state(b::GSRadials) = (poly = b.poly,)
_init_luxstate(b::GSRadials) = _static_state(b)

# ---- KA kernels (default path; CPU and GPU backends) ----

@kernel function _rnl_val_ka!(R, @Const(r), @Const(ζ), @Const(D), @Const(poly), tag)
    i, k = @index(Global, NTuple)
    K = size(ζ, 2)
    ri = r[i]
    fx = _decay(tag, ri)
    s = zero(eltype(R))
    @inbounds for m = 1:K
        s += D[k, m] * exp(-ζ[k, m] * fx)
    end
    @inbounds R[i, k] = ri^poly[k] * s
end

@kernel function _rnl_ed_ka!(R, dR, @Const(r), @Const(ζ), @Const(D), @Const(poly), tag)
    i, k = @index(Global, NTuple)
    K = size(ζ, 2)
    ri = r[i]
    fx = _decay(tag, ri)
    dfx = _ddecay(tag, ri)
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

function evaluate(basis::GSRadials, r::BATCH, ps, st)
    R = _alloc_val(basis, r, ps, st)
    backend = KA.get_backend(R)
    _rnl_val_ka!(backend)(R, r, ps.ζ, ps.D, st.poly, _decaytag(basis); ndrange = size(R))
    KA.synchronize(backend)
    return R
end

evaluate(basis::GSRadials, r::BATCH) =
        evaluate(basis, r, _static_params(basis), _static_state(basis))

function evaluate_ed(basis::GSRadials, r::BATCH, ps, st)
    R  = _alloc_val(basis, r, ps, st)
    dR = _alloc_grad(basis, r, ps, st)
    backend = KA.get_backend(R)
    _rnl_ed_ka!(backend)(R, dR, r, ps.ζ, ps.D, st.poly, _decaytag(basis); ndrange = size(R))
    KA.synchronize(backend)
    return R, dR
end

evaluate_ed(basis::GSRadials, r::BATCH) =
        evaluate_ed(basis, r, _static_params(basis), _static_state(basis))

# ---- plain forward-only reference (testing oracle) ----

function evaluate_ref(basis::GSRadials, r::AbstractVector, ps = _static_params(basis))
    ζ, D = ps.ζ, ps.D
    nRad, K = size(ζ)
    nX = length(r)
    R = zeros(promote_type(eltype(r), eltype(ζ), eltype(D)), nX, nRad)
    for k = 1:nRad, i = 1:nX
        fx = _decay(basis, r[i])
        s = zero(eltype(R))
        for m = 1:K
            s += D[k, m] * exp(-ζ[k, m] * fx)
        end
        R[i, k] = r[i]^basis.poly[k] * s
    end
    return R
end

# ---- parameter pullback ----

# TODO: move this to a KA implementation
function pullback_ps(∂R, basis::GSRadials, r::BATCH, ps, st)
    ζ, D = ps.ζ, ps.D
    nRad, K = size(ζ)
    nX = length(r)
    ∂ζ = fill!(similar(ζ), 0)
    ∂D = fill!(similar(D), 0)
    for k = 1:nRad, i = 1:nX
        fx = _decay(basis, r[i])
        rp = r[i]^basis.poly[k]
        for m = 1:K
            a = exp(-ζ[k, m] * fx)
            ∂D[k, m] += ∂R[i, k] * rp * a
            ∂ζ[k, m] += ∂R[i, k] * rp * D[k, m] * a * (-fx)
        end
    end
    return (ζ = ∂ζ, D = ∂D)
end
