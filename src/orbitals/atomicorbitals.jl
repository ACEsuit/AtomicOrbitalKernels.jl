# Atomic-orbital basis:  ϕ_{n,l,m}(𝐫) = R_{n,l}(r) · Y_{l,m}(𝐫)

const NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}

# real spherical harmonics only (SpheriCart `SolidHarmonics`)
_ylm_valtype(::SolidHarmonics, ::Type{<: SVector{3, S}}) where {S} = S
_default_ylm(L) = SolidHarmonics(L)

# the harmonics are parameter-free; their normalisation prefactors `Flm` are
# their non-trainable state.
_static_params(::SolidHarmonics) = NamedTuple()
_static_state(Ylm::SolidHarmonics) = (Flm = Ylm.Flm,)

"""
`AtomicOrbitals{TR, LEN, TY}` : a product basis
`ϕ_{n,l,m}(𝐫) = R_{n,l}(r) · Y_{l,m}(𝐫)` of a radial basis `Rnl::TR` and a real
spherical-harmonics angular basis `Ylm::TY`.

`Rnl` may be **any** basis over the scalar radius `r` exposing `natural_indices`
as `(n, l)`; `Ylm` is a SpheriCart `SolidHarmonics`. The orbital `spec` is
`(n, l, m)`, generated as `{(n,l,m) : (n,l) ∈ Rnl, m ∈ -l:l}`, so the type is
agnostic to how the radial is built. `TR` is the leading type parameter so that
families can be aliased, e.g. `GaussianTypeOrbitals = AtomicOrbitals{<:GaussianTypeRadials}`.

Evaluation (`evaluate` / `evaluate_ed`) runs through KernelAbstractions on both
CPU and GPU; `evaluate_ref` is a plain forward-only oracle. The per-orbital
radial/angular index maps live in the (non-trainable) **state**, alongside the
`Ylm` `Flm`, so they move to the device with the state:
`ps = (Rnl = (ζ, D), Ylm = (;))`, `st = (Rnl = (poly,), Ylm = (Flm,), iR, iY)`.
"""
struct AtomicOrbitals{TR, LEN, TY} <: AbstractP4MLBasis
    Rnl::TR
    Ylm::TY
    spec::SVector{LEN, NT_NLM}
    radidx::SVector{LEN, Int}    # radial-basis index per orbital
    ylmidx::SVector{LEN, Int}    # Ylm index per orbital
end

const GaussianTypeOrbitals{LEN, TY} = AtomicOrbitals{<: GaussianTypeRadials, LEN, TY}
const SlaterTypeOrbitals{LEN, TY}   = AtomicOrbitals{<: SlaterTypeRadials, LEN, TY}

function AtomicOrbitals(Rnl, Ylm)
    rad_spec = natural_indices(Rnl)            # (n, l) per radial function
    inv_Ylm  = _invmap(natural_indices(Ylm))   # (l, m) -> index
    spec   = NT_NLM[]
    radidx = Int[]
    ylmidx = Int[]
    for (k, nl) in enumerate(rad_spec)
        for m = -nl.l:nl.l
            push!(spec,   (n = nl.n, l = nl.l, m = m))
            push!(radidx, k)
            push!(ylmidx, inv_Ylm[(l = nl.l, m = m)])
        end
    end
    LEN = length(spec)
    return AtomicOrbitals{typeof(Rnl), LEN, typeof(Ylm)}(
                Rnl, Ylm, SVector{LEN, NT_NLM}(spec),
                SVector{LEN, Int}(radidx), SVector{LEN, Int}(ylmidx))
end

Base.length(basis::AtomicOrbitals) = length(basis.spec)
natural_indices(basis::AtomicOrbitals) = basis.spec
_generate_input(::AtomicOrbitals) = @SVector randn(3)
Base.show(io::IO, b::AtomicOrbitals) = print(io, "AtomicOrbitals($(b.Rnl), $(b.Ylm))")

_valtype(b::AtomicOrbitals, T::Type{<: SVector{3, S}}) where {S} =
        promote_type(_valtype(b.Rnl, S), _ylm_valtype(b.Ylm, T))
_valtype(b::AtomicOrbitals, T::Type{<: SVector{3, S}},
         ::Union{Nothing, @NamedTuple{}}, st) where {S} =
        promote_type(_valtype(b.Rnl, S), _ylm_valtype(b.Ylm, T))
_valtype(b::AtomicOrbitals, T::Type{<: SVector{3, S}}, ps, st) where {S} =
        promote_type(_valtype(b.Rnl, S, ps.Rnl, st.Rnl), _ylm_valtype(b.Ylm, T))

_static_params(b::AtomicOrbitals) =
        (Rnl = _static_params(b.Rnl), Ylm = _static_params(b.Ylm))
# index maps are non-trainable state, so they ride along with device transforms
_static_state(b::AtomicOrbitals) =
        (Rnl = _static_state(b.Rnl), Ylm = _static_state(b.Ylm),
         iR = b.radidx, iY = b.ylmidx)
_init_luxparams(rng::AbstractRNG, b::AtomicOrbitals) =
        (Rnl = _init_luxparams(rng, b.Rnl), Ylm = _init_luxparams(rng, b.Ylm))
_init_luxstate(rng::AbstractRNG, b::AtomicOrbitals) = _static_state(b)

# ---- KA evaluation (default path; CPU and GPU) ----
#
# `Rnl` and `Ylm` are independent and could in principle be evaluated
# concurrently. Right now they run serially: each sub-`evaluate` synchronises
# internally, and a single backend stream runs kernels in issue order anyway.
# True overlap needs separate streams — backend-specific (breaks the KA-only
# path) and marginal once a kernel saturates the device. The portable win is
# launch-ahead (non-syncing launchers + one sync before the product kernel);
# deferred for now.

@kernel function _aorb_val_ka!(Rnlm, @Const(Rnl), @Const(Ylm), @Const(iR), @Const(iY))
    j, i = @index(Global, NTuple)
    @inbounds Rnlm[j, i] = Rnl[j, iR[i]] * Ylm[j, iY[i]]
end

@kernel function _aorb_ed_ka!(Rnlm, dRnlm, @Const(Rnl), @Const(dRnl), @Const(Ylm),
                              @Const(dYlm), @Const(X), @Const(r), @Const(iR), @Const(iY))
    j, i = @index(Global, NTuple)
    @inbounds begin
        k = iR[i]; y = iY[i]
        rn = Rnl[j, k]; yl = Ylm[j, y]
        Rnlm[j, i] = rn * yl
        drj = X[j] / r[j]
        dRnlm[j, i] = dRnl[j, k] * drj * yl + rn * dYlm[j, y]
    end
end

# Input extraction (mirrors the radial helpers): positions from a coordinate
# vector (identity) or from `PState`s; species indices default to 1 for plain
# coordinates, or come from `x.S` for `PState`s.
_positions(Rs::AbstractVector{<:SVector{3}}) = Rs
_positions(X::AbstractVector{<:PState}) = _position.(X)
_sidx(basis::AtomicOrbitals, X::AbstractVector{<:PState}) = _species_indices(basis.Rnl, X)
_sidx(basis::AtomicOrbitals, X::AbstractVector) = _default_sidx(X)

# Two input layers: an internal layer over positions `Rs` (`SVector{3}`) + species
# indices `sidx`, and a public layer over a vector `X` of positions (→ species 1)
# or `PState`s (→ species from `x.S`). The species attribute rides inside the
# input — the orbital `spec`/index maps and the product kernels stay species-free.

function evaluate(basis::AtomicOrbitals, Rs::AbstractVector{<:SVector{3}},
                  sidx::AbstractVector{<:Integer}, ps, st)
    Rnlm = _alloc_val(basis, Rs, ps, st)
    backend = KA.get_backend(Rnlm)
    r = norm.(Rs)
    Rnl = evaluate(basis.Rnl, r, sidx, ps.Rnl, st.Rnl)
    Ylm = evaluate(basis.Ylm, Rs, ps.Ylm, st.Ylm)
    # `Rnl` and `Ylm` come from independent kernel launches; make sure both are
    # finished before the product kernel consumes them.
    KA.synchronize(backend)
    _aorb_val_ka!(backend)(Rnlm, Rnl, Ylm, st.iR, st.iY; ndrange = size(Rnlm))
    KA.synchronize(backend)
    return Rnlm
end

evaluate(basis::AtomicOrbitals, X::AbstractVector, ps, st) =
        evaluate(basis, _positions(X), _sidx(basis, X), ps, st)

evaluate(basis::AtomicOrbitals, X::AbstractVector) =
        evaluate(basis, X, _static_params(basis), _static_state(basis))

function evaluate_ed(basis::AtomicOrbitals, Rs::AbstractVector{<:SVector{3}},
                     sidx::AbstractVector{<:Integer}, ps, st)
    Rnlm  = _alloc_val(basis, Rs, ps, st)
    dRnlm = _alloc_grad(basis, Rs, ps, st)
    backend = KA.get_backend(Rnlm)
    r = norm.(Rs)
    Rnl, dRnl = evaluate_ed(basis.Rnl, r, sidx, ps.Rnl, st.Rnl)
    Ylm, dYlm = evaluate_ed(basis.Ylm, Rs, ps.Ylm, st.Ylm)
    # `Rnl` and `Ylm` come from independent kernel launches; make sure both are
    # finished before the product kernel consumes them.
    KA.synchronize(backend)
    _aorb_ed_ka!(backend)(Rnlm, dRnlm, Rnl, dRnl, Ylm, dYlm, Rs, r, st.iR, st.iY;
                          ndrange = size(Rnlm))
    KA.synchronize(backend)
    return Rnlm, dRnlm
end

evaluate_ed(basis::AtomicOrbitals, X::AbstractVector, ps, st) =
        evaluate_ed(basis, _positions(X), _sidx(basis, X), ps, st)

evaluate_ed(basis::AtomicOrbitals, X::AbstractVector) =
        evaluate_ed(basis, X, _static_params(basis), _static_state(basis))

# ---- plain forward-only reference (testing oracle) ----

function evaluate_ref(basis::AtomicOrbitals, X::AbstractVector,
                      ps = _static_params(basis), st = _static_state(basis))
    Rs = _positions(X)
    Rnl = evaluate_ref(basis.Rnl, norm.(Rs), _sidx(basis, X), ps.Rnl, st.Rnl)
    Ylm = evaluate(basis.Ylm, Rs, ps.Ylm, st.Ylm)
    return Rnl[:, basis.radidx] .* Ylm[:, basis.ylmidx]
end

# ---- parameter pullback ----

# contract the orbital cotangent back onto the radial cotangent ∂Rnl. Several
# orbitals share a radial index `iR`, so the scatter collides → atomic add. A
# future perf path is an atomic-free gather over a radial→orbital inverse map.
@kernel function _aorb_pbrad_ka!(∂Rnl, @Const(∂Rnlm), @Const(Ylm),
                                 @Const(iR), @Const(iY))
    j, i = @index(Global, NTuple)
    @inbounds KA.@atomic ∂Rnl[j, iR[i]] += ∂Rnlm[j, i] * Ylm[j, iY[i]]
end

function pullback_ps(∂Rnlm, basis::AtomicOrbitals, Rs::AbstractVector{<:SVector{3}},
                     sidx::AbstractVector{<:Integer}, ps::NamedTuple, st)
    T = promote_type(eltype(∂Rnlm), eltype(eltype(Rs)))
    r = norm.(Rs)
    Ylm = evaluate(basis.Ylm, Rs, ps.Ylm, st.Ylm)
    ∂Rnl = fill!(similar(Ylm, T, length(Rs), length(basis.Rnl)), zero(T))
    backend = KA.get_backend(∂Rnl)
    KA.synchronize(backend)
    _aorb_pbrad_ka!(backend)(∂Rnl, ∂Rnlm, Ylm, st.iR, st.iY; ndrange = size(∂Rnlm))
    KA.synchronize(backend)
    return (Rnl = pullback_ps(∂Rnl, basis.Rnl, r, sidx, ps.Rnl, st.Rnl),
            Ylm = NamedTuple())
end

pullback_ps(∂Rnlm, basis::AtomicOrbitals, X::AbstractVector, ps::NamedTuple, st) =
        pullback_ps(∂Rnlm, basis, _positions(X), _sidx(basis, X), ps, st)

# ---- ChainRules: differentiable w.r.t. positions X and params ps ----
#
# The X-pullback uses the spatial Jacobian `dRnlm` from `evaluate_ed` (forward
# mode in X; each output depends only on its own point); the param-pullback uses
# `pullback_ps`. P4ML's generic rrule (on `AbstractP4MLBasis`) recomputes ∂X with
# *static* params; this more-specific method uses the actual `ps` and avoids that.

# parallel over the point j (the large dim), serial over orbitals i (the small
# dim) — the textbook reduction layout. Benchmarked against an (i,j)-parallel
# component-atomic variant: this is as fast or faster up to ~300 orbitals and
# avoids the extra buffer + pack kernel, so we keep it.
@kernel function _aorb_pbx_ka!(∂X, @Const(∂Rnlm), @Const(dRnlm))
    j = @index(Global)
    acc = zero(eltype(∂X))
    @inbounds for i = 1:size(dRnlm, 2)
        acc += ∂Rnlm[j, i] * dRnlm[j, i]
    end
    @inbounds ∂X[j] = acc
end

function _aorb_pullback_x(∂Rnlm, dRnlm)
    S = promote_type(eltype(∂Rnlm), eltype(eltype(dRnlm)))
    ∂X = similar(dRnlm, SVector{3, S}, size(dRnlm, 1))
    backend = KA.get_backend(∂X)
    _aorb_pbx_ka!(backend)(∂X, ∂Rnlm, dRnlm; ndrange = length(∂X))
    KA.synchronize(backend)
    return ∂X
end

# The X-cotangent mirrors the input structure: a positions (`SVector{3}`) input
# yields an `SVector{3}` gradient per point, a `PState` input yields a `VState`
# (the position gradient in the `𝐫` slot — the discrete species carries none).
_xtangent(::AbstractVector{<:SVector{3}}, ∂Rs) = ∂Rs
_xtangent(::AbstractVector{<:PState}, ∂Rs) = map(g -> VState(𝐫 = g), ∂Rs)

function _aorb_rrule(basis::AtomicOrbitals, X, Rs, sidx, ps, st)
    Rnlm, dRnlm = evaluate_ed(basis, Rs, sidx, ps, st)
    function _pb(_∂)
        ∂Rnlm = unthunk(_∂)
        ∂X  = _xtangent(X, _aorb_pullback_x(∂Rnlm, dRnlm))
        ∂ps = pullback_ps(∂Rnlm, basis, Rs, sidx, ps, st)
        return (NoTangent(), NoTangent(), ∂X, ∂ps, NoTangent())
    end
    return Rnlm, _pb
end

# Dispatch on the input kind is required (not just for disambiguation): the `∂X`
# type must match the input — `SVector{3}` for positions, `VState` for `PState`.
# This also keeps us unambiguous with P4ML's generic `rrule` (typed on a vector of
# numbers/SArrays): positions are a strict subset, `PState`s are disjoint.
rrule(::typeof(evaluate), basis::AtomicOrbitals,
      X::AbstractVector{<:SVector{3}}, ps, st) =
        _aorb_rrule(basis, X, _positions(X), _sidx(basis, X), ps, st)

rrule(::typeof(evaluate), basis::AtomicOrbitals,
      X::AbstractVector{<:PState}, ps, st) =
        _aorb_rrule(basis, X, _positions(X), _sidx(basis, X), ps, st)
