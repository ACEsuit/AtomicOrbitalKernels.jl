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

function evaluate(basis::AtomicOrbitals, X::BATCH, ps, st)
    Rnlm = _alloc_val(basis, X, ps, st)
    backend = KA.get_backend(Rnlm)
    r = norm.(X)
    Rnl = evaluate(basis.Rnl, r, ps.Rnl, st.Rnl)
    Ylm = evaluate(basis.Ylm, X, ps.Ylm, st.Ylm)
    # `Rnl` and `Ylm` come from independent kernel launches; make sure both are
    # finished before the product kernel consumes them.
    KA.synchronize(backend)
    _aorb_val_ka!(backend)(Rnlm, Rnl, Ylm, st.iR, st.iY; ndrange = size(Rnlm))
    KA.synchronize(backend)
    return Rnlm
end

evaluate(basis::AtomicOrbitals, X::BATCH) =
        evaluate(basis, X, _static_params(basis), _static_state(basis))

function evaluate_ed(basis::AtomicOrbitals, X::BATCH, ps, st)
    Rnlm  = _alloc_val(basis, X, ps, st)
    dRnlm = _alloc_grad(basis, X, ps, st)
    backend = KA.get_backend(Rnlm)
    r = norm.(X)
    Rnl, dRnl = evaluate_ed(basis.Rnl, r, ps.Rnl, st.Rnl)
    Ylm, dYlm = evaluate_ed(basis.Ylm, X, ps.Ylm, st.Ylm)
    # `Rnl` and `Ylm` come from independent kernel launches; make sure both are
    # finished before the product kernel consumes them.
    KA.synchronize(backend)
    _aorb_ed_ka!(backend)(Rnlm, dRnlm, Rnl, dRnl, Ylm, dYlm, X, r, st.iR, st.iY;
                          ndrange = size(Rnlm))
    KA.synchronize(backend)
    return Rnlm, dRnlm
end

evaluate_ed(basis::AtomicOrbitals, X::BATCH) =
        evaluate_ed(basis, X, _static_params(basis), _static_state(basis))

# ---- plain forward-only reference (testing oracle) ----

function evaluate_ref(basis::AtomicOrbitals, X::AbstractVector{<: SVector{3}},
                      ps = _static_params(basis), st = _static_state(basis))
    r = norm.(X)
    Rnl = evaluate_ref(basis.Rnl, r, ps.Rnl, st.Rnl)
    Ylm = evaluate(basis.Ylm, X, ps.Ylm, st.Ylm)
    return Rnl[:, basis.radidx] .* Ylm[:, basis.ylmidx]
end

# ---- parameter pullback ----

# TODO: move this to a KA implementation
function pullback_ps(∂Rnlm, basis::AtomicOrbitals, X::AbstractVector{<: SVector{3}},
                     ps::NamedTuple, st)
    T = promote_type(eltype(∂Rnlm), eltype(eltype(X)))
    r = norm.(X)
    Rnl = evaluate(basis.Rnl, r, ps.Rnl, st.Rnl)
    Ylm = evaluate(basis.Ylm, X, ps.Ylm, st.Ylm)
    ∂Rnl = zeros(T, size(Rnl))
    nX = length(X)
    for i = 1:length(basis)
        iR = basis.radidx[i]; iY = basis.ylmidx[i]
        for j = 1:nX
            ∂Rnl[j, iR] += ∂Rnlm[j, i] * Ylm[j, iY]
        end
    end
    return (Rnl = pullback_ps(∂Rnl, basis.Rnl, r, ps.Rnl, st.Rnl), Ylm = NamedTuple())
end
