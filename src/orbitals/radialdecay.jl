# `RadialDecay` : the decay-envelope radial basis
#   Dn_k(r) = Σ_m D[k,m] exp(-ζ[k,m] f(r))
# with f a Gaussian (f(r)=r²) or Slater (f(r)=r) decay. It is one factor of a
# `SeparableRadial`; on its own it is a regular `AbstractP4MLBasis` over scalar
# inputs `r`.

abstract type AbstractDecayFunction end

const NT_NNL = NamedTuple{(:n1, :n2, :l), Tuple{Int, Int, Int}}

struct RadialDecay{LEN, TSMAT, DF<:AbstractDecayFunction} <: AbstractP4MLBasis
    ζ::TSMAT
    D::TSMAT
    decay::DF
    spec::SVector{LEN, NT_NNL}
end

struct GaussianDecay <: AbstractDecayFunction
end

struct SlaterDecay <: AbstractDecayFunction
end

# Gaussian: f(x) = x^2, df/dx = 2x
(f::GaussianDecay)(x) = x^2
df(f::GaussianDecay, x) = 2x

# Slater: f(x) = x, df/dx = 1
(f::SlaterDecay)(x) = x
df(f::SlaterDecay, x::T) where T = one(T)

"""
    construct_basis(ζ_raw, D_raw, decay, spec_list)

Construct a `RadialDecay` from raw matrix data `ζ_raw`, `D_raw` and a decay
function `decay::AbstractDecayFunction`. All input is converted to
statically-sized `SMatrix` for efficiency.
"""
function construct_basis(ζ_raw, D_raw, decay::AbstractDecayFunction, spec_list)
    LEN, K = size(ζ_raw)
    T = promote_type(eltype(ζ_raw), eltype(D_raw))
    ζ = SMatrix{LEN, K, T}(ζ_raw)
    D = SMatrix{LEN, K, T}(D_raw)
    spec = SVector{length(spec_list)}(spec_list)
    return RadialDecay{length(spec), typeof(ζ), typeof(decay)}(ζ, D, decay, spec)
end

Base.length(basis::RadialDecay) = size(basis.ζ, 1)

_valtype(::RadialDecay, T::Type{<: Number}, args...) = T

_valtype(::RadialDecay, T::Type{<: Real},
         ps, st) = promote_type(T, eltype(ps.ζ), eltype(ps.D))

_static_params(basis::RadialDecay) = (ζ = basis.ζ, D = basis.D)

_init_luxparams(basis::RadialDecay) =
            ( ζ = Matrix(basis.ζ), D = Matrix(basis.D) )

_evaluate!(P, dP, basis::RadialDecay, x)  =
    _evaluate!(P, dP, basis, x, _static_params(basis), nothing)

natural_indices(basis::RadialDecay) = basis.spec

function _evaluate!(P, dP, basis::RadialDecay, x::AbstractVector, ps, st)
    ζ, D = ps.ζ, ps.D
    N, K = size(ζ)
    nX = length(x)
    WITHGRAD = !isnothing(dP)

    fill!(P, zero(eltype(P)))
    if WITHGRAD
        fill!(dP, zero(eltype(dP)))
    end

    decay = basis.decay

    @inbounds begin
        for n = 1:N, m = 1:K
            @simd ivdep for i = 1:nX
                fx = decay(x[i])
                a = D[n, m] * exp(-ζ[n, m] * fx)
                P[i, n] += a
                if WITHGRAD
                    dfx = df(decay, x[i])
                    dP[i, n] += -ζ[n, m] * dfx * a
                end
            end
        end
    end

    return nothing
end

function pullback_ps(∂P, basis::RadialDecay, x::BATCH, ps, st)
    ζ, D = ps.ζ, ps.D
    decay = basis.decay
    N, K = size(ζ)
    nX = length(x)

    ∂ζ = fill!(similar(ζ), 0)
    ∂D = fill!(similar(D), 0)

    @inbounds for n = 1:N, m = 1:K
        @simd ivdep for j = 1:nX
            fx  = decay(x[j])           # f(x[j])
            dfx = df(decay, x[j])       # df/dx
            a1  = exp(-ζ[n, m] * fx)
            ∂ζ[n, m] += ∂P[j, n] * D[n, m] * a1 * (-fx)
            ∂D[n, m] += ∂P[j, n] * a1
        end
    end
    return (ζ = ∂ζ, D = ∂D)
end
