# `SeparableRadial` : a radial basis whose functions are products of a
# polynomial radial part `Pn` and a decay envelope `Dn` (a `RadialDecay`),
#
#   R_k(r) = Pn_{p(k)}(r) · Dn_{d(k)}(r) .
#
# This is the `Rnl` radial consumed by `AtomicOrbitals`. The `Pn·Dn`
# factorisation is an internal evaluation/sharing detail: `Pn` is evaluated once
# and reused across all radial functions that share it. Other radial families
# (contracted Gaussians/Slaters from a basis set, splines, …) would be separate
# concrete radial types plugged into `AtomicOrbitals` in the same way; an
# abstract radial supertype is deferred until a second such type needs it.

struct SeparableRadial{LEN, TP, TD} <: AbstractP4MLBasis
    Pn::TP                              # polynomial radial part
    Dn::TD                              # decay envelope (a RadialDecay)
    spec::SVector{LEN, NT_NNL}          # (n1,n2,l) label per radial function
    specidx::Vector{Tuple{Int, Int}}    # (Pn index, Dn index) per radial function
end

function _invmap(a::AbstractVector)
    inva = Dict{eltype(a), Int}()
    for i = 1:length(a)
       inva[a[i]] = i
    end
    return inva
end

function SeparableRadial(Pn, Dn)
    spec = natural_indices(Dn)
    LEN = length(spec)
    inv_Pn = _invmap(natural_indices(Pn))
    specidx = [ (inv_Pn[(n = s.n1,)], k) for (k, s) in enumerate(spec) ]
    return SeparableRadial{LEN, typeof(Pn), typeof(Dn)}(
                Pn, Dn, SVector{LEN, NT_NNL}(spec), specidx)
end

Base.length(basis::SeparableRadial) = length(basis.spec)

natural_indices(basis::SeparableRadial) = basis.spec

_generate_input(::SeparableRadial) = rand()

_valtype(basis::SeparableRadial, T::Type{<: Number}) =
        promote_type(_valtype(basis.Pn, T), _valtype(basis.Dn, T))

_valtype(basis::SeparableRadial, T::Type{<: Number},
         ps::Union{Nothing, @NamedTuple{}}, st) =
        promote_type(_valtype(basis.Pn, T), _valtype(basis.Dn, T))

_valtype(basis::SeparableRadial, T::Type{<: Number}, ps, st) =
        promote_type(_valtype(basis.Pn, T),
                     _valtype(basis.Dn, T, ps.Dn, st.Dn))

_static_params(basis::SeparableRadial) =
        (Pn = _static_params(basis.Pn), Dn = _static_params(basis.Dn))

_init_luxparams(rng::AbstractRNG, l::SeparableRadial) =
        (Pn = _init_luxparams(rng, l.Pn), Dn = _init_luxparams(rng, l.Dn))

_init_luxstate(rng::AbstractRNG, l::SeparableRadial) =
        (Pn = _init_luxstate(rng, l.Pn), Dn = _init_luxstate(rng, l.Dn))

_evaluate!(R, dR, basis::SeparableRadial, r) =
        _evaluate!(R, dR, basis, r, _static_params(basis), (Pn = nothing, Dn = nothing))

function _evaluate!(R, dR, basis::SeparableRadial, r::AbstractVector, ps, st)
    nX = length(r)
    WITHGRAD = !isnothing(dR)

    fill!(R, zero(eltype(R)))
    WITHGRAD && fill!(dR, zero(eltype(dR)))

    if WITHGRAD
        Pn, dPn = evaluate_ed(basis.Pn, r, ps.Pn, st.Pn)
        Dn, dDn = evaluate_ed(basis.Dn, r, ps.Dn, st.Dn)
    else
        Pn = evaluate(basis.Pn, r, ps.Pn, st.Pn)
        Dn = evaluate(basis.Dn, r, ps.Dn, st.Dn)
    end

    for (k, (p, d)) in enumerate(basis.specidx)
        @simd ivdep for i = 1:nX
            R[i, k] = Pn[i, p] * Dn[i, d]
            if WITHGRAD
                dR[i, k] = dPn[i, p] * Dn[i, d] + Pn[i, p] * dDn[i, d]
            end
        end
    end

    return nothing
end

function pullback_ps(∂R, basis::SeparableRadial, r::BATCH, ps, st)
    T = promote_type(eltype(∂R), eltype(r))
    Pn = evaluate(basis.Pn, r, ps.Pn, st.Pn)
    Dn = evaluate(basis.Dn, r, ps.Dn, st.Dn)
    ∂Pn = zeros(T, size(Pn))
    ∂Dn = zeros(T, size(Dn))
    nX = length(r)

    for (k, (p, d)) in enumerate(basis.specidx)
        @simd ivdep for i = 1:nX
            ∂Pn[i, p] += ∂R[i, k] * Dn[i, d]
            ∂Dn[i, d] += ∂R[i, k] * Pn[i, p]
        end
    end

    return (Pn = pullback_ps(∂Pn, basis.Pn, r, ps.Pn, st.Pn),
            Dn = pullback_ps(∂Dn, basis.Dn, r, ps.Dn, st.Dn))
end
