# Atomic-orbital basis:  ϕ_{n,l,m}(𝐫) = R_{n,l}(r) · Ylm_{l,m}(𝐫)
#
# A product of a radial basis `Rnl` (any `AbstractP4MLBasis` over the scalar
# radius r — e.g. a `SeparableRadial`) and an angular basis `Ylm` (a SpheriCart
# harmonics basis, used through the ACEbase `evaluate` interface and carrying no
# parameters/state). Moved from Polynomials4ML v0.6.1 and restructured from the
# original `Pn·Dn·Ylm` form into this `Rnl·Ylm` form, so the radial family is a
# pluggable, independently-testable piece.

const NT_NNLM = NamedTuple{(:n1, :n2, :l, :m), Tuple{Int, Int, Int, Int}}

# angular `Ylm` value type: real fallback; SpheriCart complex harmonics below.
_ylm_valtype(Ylm, ::Type{<: SVector{3, S}}) where {S} = S
_ylm_valtype(::Union{ComplexSolidHarmonics, ComplexSphericalHarmonics},
             ::Type{<: SVector{3, S}}) where {S} = Complex{S}

# default angular basis used by the `_rand_*` example bases
_default_ylm(L) = SolidHarmonics(L)

mutable struct AtomicOrbitals{LEN, TR, TY} <: AbstractP4MLBasis
   Rnl::TR
   Ylm::TY
   spec::SVector{LEN, NT_NNLM}
   specidx::Vector{Tuple{Int, Int}}    # (radial index, Ylm index) per orbital
end

function AtomicOrbitals(Rnl, Ylm, spec::AbstractVector{NT_NNLM})
    LEN = length(spec)
    inv_Rnl = _invmap(natural_indices(Rnl))
    inv_Ylm = _invmap(natural_indices(Ylm))
    specidx = [ (inv_Rnl[(n1 = s.n1, n2 = s.n2, l = s.l)], inv_Ylm[(l = s.l, m = s.m)])
                for s in spec ]
    return AtomicOrbitals{LEN, typeof(Rnl), typeof(Ylm)}(
                Rnl, Ylm, SVector{LEN, NT_NNLM}(spec), specidx)
end

Base.length(basis::AtomicOrbitals) = length(basis.spec)

natural_indices(basis::AtomicOrbitals) = basis.spec

_generate_input(basis::AtomicOrbitals) = @SVector randn(3)

Base.show(io::IO, basis::AtomicOrbitals) =
        print(io, "AtomicOrbitals($(basis.Rnl), $(basis.Ylm))")

_valtype(basis::AtomicOrbitals, T::Type{<: SVector{3, S}}) where {S} =
        promote_type(_valtype(basis.Rnl, S), _ylm_valtype(basis.Ylm, T))

_valtype(basis::AtomicOrbitals, T::Type{<: SVector{3, S}},
            ps::Union{Nothing, @NamedTuple{}}, st) where {S} =
        promote_type(_valtype(basis.Rnl, S), _ylm_valtype(basis.Ylm, T))

_valtype(basis::AtomicOrbitals, T::Type{<: SVector{3, S}}, ps, st) where {S} =
        promote_type(_valtype(basis.Rnl, S, ps.Rnl, st.Rnl),
                     _ylm_valtype(basis.Ylm, T))

# `Ylm` carries no P4ML parameters/state.
_static_params(basis::AtomicOrbitals) =
        (Rnl = _static_params(basis.Rnl), Ylm = NamedTuple())

_init_luxparams(rng::AbstractRNG, l::AtomicOrbitals) =
        (Rnl = _init_luxparams(rng, l.Rnl), Ylm = NamedTuple())

_init_luxstate(rng::AbstractRNG, l::AtomicOrbitals) =
        (Rnl = _init_luxstate(rng, l.Rnl), Ylm = NamedTuple())

# -------- Evaluation (Rnl · Ylm; Bumper/WithAlloc-free) --------

_evaluate!(Rnlm, dRnlm, basis::AtomicOrbitals, X) =
            _evaluate!(Rnlm, dRnlm, basis, X, _static_params(basis),
                       (Rnl = (Pn = nothing, Dn = nothing), Ylm = nothing))

function _evaluate!(Rnlm, dRnlm, basis::AtomicOrbitals,
                    X::AbstractVector{<: SVector{3}}, ps, st)
    nX = length(X)
    WITHGRAD = !isnothing(dRnlm)

    fill!(Rnlm, zero(eltype(Rnlm)))
    WITHGRAD && fill!(dRnlm, zero(eltype(dRnlm)))

    r = map(norm, X)

    if WITHGRAD
        Rn, dRn = evaluate_ed(basis.Rnl, r, ps.Rnl, st.Rnl)
        Ylm, dYlm = evaluate_ed(basis.Ylm, X)
    else
        Rn = evaluate(basis.Rnl, r, ps.Rnl, st.Rnl)
        Ylm = evaluate(basis.Ylm, X)
    end

    for (i, (k, y)) in enumerate(basis.specidx)
        @simd ivdep for j = 1:nX
            Rnlm[j, i] = Rn[j, k] * Ylm[j, y]
            if WITHGRAD
                drj = X[j] / r[j]
                dRnlm[j, i] = dRn[j, k] * drj * Ylm[j, y] + Rn[j, k] * dYlm[j, y]
            end
        end
    end

    return nothing
end

function pullback_ps(∂Rnlm, basis::AtomicOrbitals,
                     X::AbstractVector{<: SVector{3}}, ps::NamedTuple, st)
    T = promote_type(eltype(∂Rnlm), eltype(eltype(X)))
    r = map(norm, X)
    Rn = evaluate(basis.Rnl, r, ps.Rnl, st.Rnl)
    Ylm = evaluate(basis.Ylm, X)   # angular basis, param-free
    ∂Rn = zeros(T, size(Rn))
    nX = length(X)

    for (i, (k, y)) in enumerate(basis.specidx)
        @simd ivdep for j = 1:nX
            ∂Rn[j, k] += ∂Rnlm[j, i] * Ylm[j, y]
        end
    end

    return (Rnl = pullback_ps(∂Rn, basis.Rnl, r, ps.Rnl, st.Rnl),
            Ylm = NamedTuple())
end

# -------- ready-made example/test bases --------

# `_rand_*` build example `AtomicOrbitals` over a Gaussian/Slater `SeparableRadial`
# and a SpheriCart `SolidHarmonics` angular basis.
function _rand_basis(N1=4, N2=3;
    K::Int=1,
    T::Type=Float64,
    decay_type::AbstractDecayFunction=GaussianDecay(),
    ζinit = (n, k) -> rand(T, n, k),
    Dinit = (n, k) -> ones(T, n, k))

    Pn = MonoBasis(N1 + 1)
    Ylm = _default_ylm(N1 - 1)
    spec_list = [(n1=n1, n2=n2, l=l, m=m)
                 for n1 in 1:N1, n2 in 1:N2, l in 0:N1-1 for m in -l:l]
    spec = SVector{length(spec_list)}(spec_list)
    spec_ln = collect(unique((n1=s.n1, n2=s.n2, l=s.l) for s in spec))
    nln = length(spec_ln)
    Dn = construct_basis(ζinit(nln, K), Dinit(nln, K), decay_type, spec_ln)
    Rnl = SeparableRadial(Pn, Dn)
    return AtomicOrbitals(Rnl, Ylm, spec)
end

_rand_gaussian_basis(N1=4, N2=3, T=Float64) = _rand_basis(N1, N2; T=T)

_rand_slater_basis(N1=4, N2=3, T=Float64) =
        _rand_basis(N1, N2; T=T, decay_type = SlaterDecay())

_rand_sto_basis(N1=4, N2=2, K=4, T=Float64) = _rand_basis(N1, N2; T=T, K=K,
        ζinit = (n, k) -> rand(T, n, k),
        Dinit = (n, k) -> rand(T, n, k))
