# Atomic-orbital basis:
#   ϕ_{n1,n2,l,m}(𝐫) = Pn_{n1}(r) · Dn_{n2,l}(r) · Ylm_{l,m}(𝐫)
#
# Moved from Polynomials4ML v0.6.1 (src/atomicorbitals/), to be removed from
# Polynomials4ML. The types are concrete and built on the P4ML
# `AbstractP4MLBasis` interface; P4ML remains a dependency for the radial
# building blocks (`MonoBasis`) and the evaluation/AD/Lux machinery, and
# SpheriCart supplies the angular `Ylm`. The Bumper/WithAlloc CPU fast-path of
# the original `_evaluate!` is dropped in favour of plain allocation; GPU-first
# KA evaluation is a planned follow-up.

abstract type AbstractDecayFunction end

const NT_NNL = NamedTuple{(:n1, :n2, :l), Tuple{Int, Int, Int}}
const NT_NNLM = NamedTuple{(:n1, :n2, :l, :m), Tuple{Int, Int, Int, Int}}

struct RadialDecay{LEN, TSMAT, DF<:AbstractDecayFunction} <: AbstractP4MLBasis
    ζ::TSMAT
    D::TSMAT
    decay::DF
    spec::SVector{LEN, NT_NNL}
end

"""
`AtomicOrbitals` : a quantum-chemistry atomic-orbital basis whose functions are
products `ϕ_{n1,n2,l,m}(𝐫) = Pn_{n1}(r) * Dn_{n2,l}(r) * Ylm_{l,m}(𝐫)` of a
radial polynomial part `Pn`, a radial decay part `Dn` (a `RadialDecay`), and an
angular part `Ylm` (a SpheriCart harmonics basis, used purely through the
ACEbase `evaluate` interface and carrying no parameters/state).
"""
mutable struct AtomicOrbitals{LEN, TP, TD, TY}  <: AbstractP4MLBasis
   Pn::TP
   Dn::TD
   Ylm::TY
   spec::SVector{LEN, NT_NNLM}
   specidx::Vector{Tuple{Int64, Int64, Int64}}
end

function AtomicOrbitals(Pn, Dn, Ylm, spec::AbstractVector{NT_NNLM}, specidx)
    LEN = length(spec)
    return AtomicOrbitals{LEN, typeof(Pn), typeof(Dn), typeof(Ylm)}(
                Pn, Dn, Ylm, SVector{LEN, NT_NNLM}(spec), specidx)
end

Base.length(basis::AtomicOrbitals) = length(basis.spec)

natural_indices(basis::AtomicOrbitals) = basis.spec

# angular `Ylm` value type: real fallback; the SpheriCart complex harmonics are
# specialised below.
_ylm_valtype(Ylm, ::Type{<: SVector{3, S}}) where {S} = S
_ylm_valtype(::Union{ComplexSolidHarmonics, ComplexSphericalHarmonics},
             ::Type{<: SVector{3, S}}) where {S} = Complex{S}

# default angular basis used by the `_rand_*` example bases
_default_ylm(L) = SolidHarmonics(L)

_valtype(basis::AtomicOrbitals, T::Type{<: SVector{3, S}}) where {S} =
        promote_type(_valtype(basis.Pn, S), _valtype(basis.Dn, S),
                     _ylm_valtype(basis.Ylm, T))

_valtype(basis::AtomicOrbitals, T::Type{<: SVector{3, S}},
            ps::Union{Nothing, @NamedTuple{}}, st) where {S} =
        promote_type(_valtype(basis.Pn, S), _valtype(basis.Dn, S),
                     _ylm_valtype(basis.Ylm, T))

_valtype(basis::AtomicOrbitals, T::Type{<: SVector{3, S}}, ps, st) where {S} =
        promote_type(_valtype(basis.Pn, S, ps.Dn, st.Dn),
                     _valtype(basis.Dn, S, ps.Dn, st.Dn),
                     _ylm_valtype(basis.Ylm, T))

_generate_input(basis::AtomicOrbitals) = @SVector randn(3)

Base.show(io::IO, basis::AtomicOrbitals) =
        print(io, "AtomicOrbitals($(basis.Pn), $(typeof(basis.Dn.decay).name.name), $(basis.Ylm))")

include("radialdecay.jl")

# `_static_params` extracts the internally-stored parameters (parameter-free
# convention); `_init_luxparams`/`_init_luxstate` initialise them Lux-style.
# `Ylm` carries no P4ML parameters/state.

_static_params(basis::AtomicOrbitals) =
        (Pn = _static_params(basis.Pn), Dn = _static_params(basis.Dn),
         Ylm = NamedTuple())

_init_luxparams(rng::AbstractRNG, l::AtomicOrbitals) =
        ( Pn = _init_luxparams(rng, l.Pn),
          Dn = _init_luxparams(rng, l.Dn),
          Ylm = NamedTuple())

_init_luxstate(rng::AbstractRNG, l::AtomicOrbitals) =
        ( Pn = _init_luxstate(rng, l.Pn),
          Dn = _init_luxstate(rng, l.Dn),
          Ylm = NamedTuple())

# -------- Evaluation (Bumper/WithAlloc removed; plain allocation) --------

_evaluate!(Rnlm, dRnlm, basis::AtomicOrbitals, X) =
            _evaluate!(Rnlm, dRnlm, basis, X,
                       _static_params(basis),
                       (Pn = nothing, Dn = nothing, Ylm = nothing))

function _evaluate!(Rnl, dRnl, basis::AtomicOrbitals,
                    X::AbstractVector{<: SVector{3}}, ps, st)
    nR = length(X)
    WITHGRAD = !isnothing(dRnl)

    fill!(Rnl, zero(eltype(Rnl)))
    WITHGRAD && fill!(dRnl, zero(eltype(dRnl)))

    R = map(norm, X)

    if WITHGRAD
        Pn, dPn = evaluate_ed(basis.Pn, R, ps.Pn, st.Pn)
        Dn, dDn = evaluate_ed(basis.Dn, R, ps.Dn, st.Dn)
        Ylm, dYlm = evaluate_ed(basis.Ylm, X)
    else
        Pn = evaluate(basis.Pn, R, ps.Pn, st.Pn)
        Dn = evaluate(basis.Dn, R, ps.Dn, st.Dn)
        Ylm = evaluate(basis.Ylm, X)
        dPn = dDn = dYlm = nothing
    end

    for (i, b) in enumerate(basis.specidx)
        @simd ivdep for j = 1:nR
            Rnl[j, i] = Pn[j, b[1]] * Dn[j, b[2]] * Ylm[j, b[3]]
            if WITHGRAD
                drj = X[j] / R[j]
                dRnl[j, i] = ( dPn[j, b[1]] * drj * Dn[j, b[2]] * Ylm[j, b[3]] +
                                Pn[j, b[1]] * dDn[j, b[2]] * drj * Ylm[j, b[3]] +
                                Pn[j, b[1]] * Dn[j, b[2]] * dYlm[j, b[3]] )
            end
        end
    end

    return nothing
end

function pullback_ps(∂Rnl, basis::AtomicOrbitals,
                     X::AbstractVector{<: SVector{3}}, ps::NamedTuple, st)
    TR = eltype(eltype(X))
    T = promote_type(eltype(∂Rnl), TR)
    nR = length(X)
    R = zeros(T, nR)
    map!(norm, R, X)

    Pn = evaluate(basis.Pn, R, ps.Pn, st.Pn)
    Dn = evaluate(basis.Dn, R, ps.Dn, st.Dn)
    Ylm = evaluate(basis.Ylm, X)   # angular basis, param-free
    ∂Pn = zeros(T, size(Pn))
    ∂Dn = zeros(T, size(Dn))

    for (i, b) in enumerate(basis.specidx)
        @simd ivdep for j = 1:nR
            ∂Pn[j, b[1]] += ∂Rnl[j, i] * Dn[j, b[2]] * Ylm[j, b[3]]
            ∂Dn[j, b[2]] += ∂Rnl[j, i] * Pn[j, b[1]] * Ylm[j, b[3]]
        end
    end

    ∂p_Pn = pullback_ps(∂Pn, basis.Pn, R, ps.Pn, st.Pn)
    ∂p_Dn = pullback_ps(∂Dn, basis.Dn, R, ps.Dn, st.Dn)
    # Ylm has no parameters
    return (Pn = ∂p_Pn, Dn = ∂p_Dn, Ylm = NamedTuple())
end
