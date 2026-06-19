# Atomic-orbital basis:  ϕ_{n,l,m}(𝐫) = R_{n,l}(r) · Y_{l,m}(𝐫)

const NT_NLM = NamedTuple{(:n, :l, :m), Tuple{Int, Int, Int}}

# real spherical harmonics only (SpheriCart `SolidHarmonics`)
_ylm_valtype(::SolidHarmonics, ::Type{<: SVector{3, S}}) where {S} = S
_default_ylm(L) = SolidHarmonics(L)

# the harmonics' normalisation prefactors `Flm` are their (non-trainable) state
_static_state(Ylm::SolidHarmonics) = (Flm = Ylm.Flm,)

"""
`AtomicOrbitals` : a product basis `ϕ_{n,l,m}(𝐫) = R_{n,l}(r) · Y_{l,m}(𝐫)` of a
radial basis `Rnl` and a real spherical-harmonics angular basis `Ylm`.

`Rnl` may be **any** basis over the scalar radius `r` that exposes
`natural_indices` as `(n, l)` (e.g. [`Rnl`](@ref)); `Ylm` is a SpheriCart
`SolidHarmonics`. The orbital `spec` is `(n, l, m)` and is generated as
`{(n,l,m) : (n,l) ∈ Rnl, m ∈ -l:l}`, so the type is agnostic to how the radial
is built. Evaluation (`evaluate` / `evaluate_ed`) runs through
KernelAbstractions on both CPU and GPU; `evaluate_ref` is a plain forward-only
oracle for testing. Learnable parameters live in `Rnl`; `ps = (Rnl = …, Ylm =
(;))`, `st = (Rnl = …, Ylm = (Flm = …,))`.
"""
mutable struct AtomicOrbitals{LEN, TR, TY} <: AbstractP4MLBasis
    Rnl::TR
    Ylm::TY
    spec::SVector{LEN, NT_NLM}
    radidx::Vector{Int}    # radial-basis index per orbital
    ylmidx::Vector{Int}    # Ylm index per orbital
end

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
    return AtomicOrbitals{LEN, typeof(Rnl), typeof(Ylm)}(
                Rnl, Ylm, SVector{LEN, NT_NLM}(spec), radidx, ylmidx)
end

Base.length(basis::AtomicOrbitals) = length(basis.spec)
natural_indices(basis::AtomicOrbitals) = basis.spec
_generate_input(::AtomicOrbitals) = @SVector randn(3)
Base.show(io::IO, b::AtomicOrbitals) =
        print(io, "AtomicOrbitals($(b.Rnl), $(b.Ylm))")

_valtype(b::AtomicOrbitals, T::Type{<: SVector{3, S}}) where {S} =
        promote_type(_valtype(b.Rnl, S), _ylm_valtype(b.Ylm, T))
_valtype(b::AtomicOrbitals, T::Type{<: SVector{3, S}},
         ::Union{Nothing, @NamedTuple{}}, st) where {S} =
        promote_type(_valtype(b.Rnl, S), _ylm_valtype(b.Ylm, T))
_valtype(b::AtomicOrbitals, T::Type{<: SVector{3, S}}, ps, st) where {S} =
        promote_type(_valtype(b.Rnl, S, ps.Rnl, st.Rnl), _ylm_valtype(b.Ylm, T))

_static_params(b::AtomicOrbitals) = (Rnl = _static_params(b.Rnl), Ylm = NamedTuple())
_static_state(b::AtomicOrbitals)  = (Rnl = _static_state(b.Rnl),  Ylm = _static_state(b.Ylm))
_init_luxparams(rng::AbstractRNG, b::AtomicOrbitals) =
        (Rnl = _init_luxparams(rng, b.Rnl), Ylm = NamedTuple())
_init_luxstate(rng::AbstractRNG, b::AtomicOrbitals) =
        (Rnl = _init_luxstate(rng, b.Rnl), Ylm = _static_state(b.Ylm))

# ---- KA evaluation (default path; CPU and GPU) ----

@kernel function _aorb_val_ka!(Rnlm, @Const(Rn), @Const(Ylm), @Const(kk), @Const(yy))
    j, i = @index(Global, NTuple)
    @inbounds Rnlm[j, i] = Rn[j, kk[i]] * Ylm[j, yy[i]]
end

@kernel function _aorb_ed_ka!(Rnlm, dRnlm, @Const(Rn), @Const(dRn), @Const(Ylm),
                              @Const(dYlm), @Const(X), @Const(r), @Const(kk), @Const(yy))
    j, i = @index(Global, NTuple)
    @inbounds begin
        k = kk[i]; y = yy[i]
        rn = Rn[j, k]; yl = Ylm[j, y]
        Rnlm[j, i] = rn * yl
        drj = X[j] / r[j]
        dRnlm[j, i] = dRn[j, k] * drj * yl + rn * dYlm[j, y]
    end
end

function evaluate(basis::AtomicOrbitals, X::BATCH, ps, st)
    Rnlm = _alloc_val(basis, X, ps, st)
    backend = KA.get_backend(Rnlm)
    r = norm.(X)
    Rn = evaluate(basis.Rnl, r, ps.Rnl, st.Rnl)
    Ylm = evaluate(basis.Ylm, X)
    kk = _device_like(Rnlm, basis.radidx)
    yy = _device_like(Rnlm, basis.ylmidx)
    _aorb_val_ka!(backend)(Rnlm, Rn, Ylm, kk, yy; ndrange = size(Rnlm))
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
    Rn, dRn = evaluate_ed(basis.Rnl, r, ps.Rnl, st.Rnl)
    Ylm, dYlm = evaluate_ed(basis.Ylm, X)
    kk = _device_like(Rnlm, basis.radidx)
    yy = _device_like(Rnlm, basis.ylmidx)
    _aorb_ed_ka!(backend)(Rnlm, dRnlm, Rn, dRn, Ylm, dYlm, X, r, kk, yy;
                          ndrange = size(Rnlm))
    KA.synchronize(backend)
    return Rnlm, dRnlm
end

evaluate_ed(basis::AtomicOrbitals, X::BATCH) =
        evaluate_ed(basis, X, _static_params(basis), _static_state(basis))

# ---- plain forward-only reference (testing oracle) ----

function evaluate_ref(basis::AtomicOrbitals, X::AbstractVector{<: SVector{3}},
                      ps = _static_params(basis))
    r = norm.(X)
    Rn = evaluate_ref(basis.Rnl, r, ps.Rnl)
    Ylm = evaluate(basis.Ylm, X)
    T = promote_type(eltype(Rn), eltype(Ylm))
    nX = length(X)
    Rnlm = zeros(T, nX, length(basis))
    for i = 1:length(basis)
        k = basis.radidx[i]; y = basis.ylmidx[i]
        for j = 1:nX
            Rnlm[j, i] = Rn[j, k] * Ylm[j, y]
        end
    end
    return Rnlm
end

# ---- parameter pullback ----

function pullback_ps(∂Rnlm, basis::AtomicOrbitals, X::AbstractVector{<: SVector{3}},
                     ps::NamedTuple, st)
    T = promote_type(eltype(∂Rnlm), eltype(eltype(X)))
    r = norm.(X)
    Rn = evaluate(basis.Rnl, r, ps.Rnl, st.Rnl)
    Ylm = evaluate(basis.Ylm, X)
    ∂Rn = zeros(T, size(Rn))
    nX = length(X)
    for i = 1:length(basis)
        k = basis.radidx[i]; y = basis.ylmidx[i]
        for j = 1:nX
            ∂Rn[j, k] += ∂Rnlm[j, i] * Ylm[j, y]
        end
    end
    return (Rnl = pullback_ps(∂Rn, basis.Rnl, r, ps.Rnl, st.Rnl), Ylm = NamedTuple())
end

# ---- concrete example bases (deterministic) ----

# build a radial whose functions are enumerated `(n, l)`: for each l in 0:N1-1,
# the N1*N2 pairs (n1, n2) give the polynomial degree (n1-1) and a deterministic
# exponent/coefficient set.
function _example_rnl(N1, N2; K::Int = 1, T::Type = Float64,
                      decay::AbstractDecayFunction = GaussianDecay())
    spec  = NT_NL[]
    poly  = Int[]
    ζrows = Vector{T}[]
    Drows = Vector{T}[]
    for l = 0:N1-1
        n = 0
        for n1 = 1:N1, n2 = 1:N2
            n += 1
            push!(spec, (n = n, l = l))
            push!(poly, n1 - 1)
            push!(ζrows, T[ T(0.5) * (n2 + j) + l for j = 1:K ])
            push!(Drows, decay isa GaussianDecay ? ones(T, K) : T[ 1 / j for j = 1:K ])
        end
    end
    ζ = permutedims(reduce(hcat, ζrows))   # [nRad × K]
    D = permutedims(reduce(hcat, Drows))
    return Rnl(ζ, D, poly, decay, spec)
end

"""`gaussian_orbitals(N1=4, N2=3; T=Float64)` : a deterministic example Gaussian
atomic-orbital basis."""
gaussian_orbitals(N1 = 4, N2 = 3; T = Float64) =
        AtomicOrbitals(_example_rnl(N1, N2; T = T, decay = GaussianDecay()),
                       _default_ylm(N1 - 1))

"""`slater_orbitals(N1=4, N2=3; T=Float64)` : a deterministic example Slater
atomic-orbital basis."""
slater_orbitals(N1 = 4, N2 = 3; T = Float64) =
        AtomicOrbitals(_example_rnl(N1, N2; T = T, decay = SlaterDecay()),
                       _default_ylm(N1 - 1))

"""`sto_orbitals(N1=4, N2=2; K=4, T=Float64)` : a deterministic example
Slater-type-orbital basis (Slater decay with `K` contraction terms)."""
sto_orbitals(N1 = 4, N2 = 2; K::Int = 4, T = Float64) =
        AtomicOrbitals(_example_rnl(N1, N2; K = K, T = T, decay = SlaterDecay()),
                       _default_ylm(N1 - 1))
