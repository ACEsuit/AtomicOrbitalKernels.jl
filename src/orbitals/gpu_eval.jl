# GPU-first KernelAbstractions evaluation for the orbital types.
#
# Two nested 2-factor assemblies:
#   SeparableRadial:  R[i,k]   = Pn[i,p] · Dn[i,d]
#   AtomicOrbitals:   ϕ[j,i]   = Rnl[j,k] · Ylm[j,y]
# `Pn` (MonoBasis) and `Ylm` (SpheriCart) already have GPU evaluation paths; the
# `RadialDecay` envelope and the two assemblies are the kernels below. Radial
# parameters come from the Lux `ps`, which on the GPU live on the device. These
# kernels run on any KA backend, so the same path also runs on the CPU backend.

# ---- decay envelope  Dn(r) = Σ_m D[n,m] exp(-ζ[n,m] f(r)) ----

@kernel function _radial_val_ka!(P, @Const(x), @Const(ζ), @Const(D), decay)
    i, n = @index(Global, NTuple)
    K = size(ζ, 2)
    xi = x[i]
    fx = decay(xi)
    acc = zero(eltype(P))
    @inbounds for m = 1:K
        acc += D[n, m] * exp(-ζ[n, m] * fx)
    end
    @inbounds P[i, n] = acc
end

@kernel function _radial_ed_ka!(P, dP, @Const(x), @Const(ζ), @Const(D), decay)
    i, n = @index(Global, NTuple)
    K = size(ζ, 2)
    xi = x[i]
    fx = decay(xi)
    dfx = df(decay, xi)
    acc = zero(eltype(P))
    dacc = zero(eltype(dP))
    @inbounds for m = 1:K
        a = D[n, m] * exp(-ζ[n, m] * fx)
        acc += a
        dacc += -ζ[n, m] * dfx * a
    end
    @inbounds P[i, n] = acc
    @inbounds dP[i, n] = dacc
end

function _decay_eval_gpu(basis::RadialDecay, x, ps; withgrad::Bool)
    ζ, D = ps.ζ, ps.D
    N = size(ζ, 1)
    nX = length(x)
    backend = KA.get_backend(x)
    T = promote_type(eltype(x), eltype(ζ), eltype(D))
    P = similar(x, T, nX, N)
    if withgrad
        dP = similar(x, T, nX, N)
        _radial_ed_ka!(backend)(P, dP, x, ζ, D, basis.decay; ndrange = (nX, N))
        return P, dP
    else
        _radial_val_ka!(backend)(P, x, ζ, D, basis.decay; ndrange = (nX, N))
        return P
    end
end

# copy a host index vector to a device vector matching `ref`'s backend
_device_like(ref, v::AbstractVector) =
        copyto!(similar(ref, eltype(v), length(v)), v)

# ---- SeparableRadial assembly  R[i,k] = Pn[i,p]·Dn[i,d] ----

@kernel function _sep_val_ka!(R, @Const(Pn), @Const(Dn), @Const(p), @Const(d))
    i, k = @index(Global, NTuple)
    @inbounds R[i, k] = Pn[i, p[k]] * Dn[i, d[k]]
end

@kernel function _sep_ed_ka!(R, dR, @Const(Pn), @Const(dPn), @Const(Dn), @Const(dDn),
                             @Const(p), @Const(d))
    i, k = @index(Global, NTuple)
    @inbounds begin
        pp = p[k]; dd = d[k]
        R[i, k] = Pn[i, pp] * Dn[i, dd]
        dR[i, k] = dPn[i, pp] * Dn[i, dd] + Pn[i, pp] * dDn[i, dd]
    end
end

function evaluate!(R::AbstractGPUArray, basis::SeparableRadial, r::BATCH, ps, st)
    Pn = evaluate(basis.Pn, r)                              # MonoBasis GPU, param-free
    Dn = _decay_eval_gpu(basis.Dn, r, ps.Dn; withgrad = false)
    p = _device_like(R, [t[1] for t in basis.specidx])
    d = _device_like(R, [t[2] for t in basis.specidx])
    backend = KA.get_backend(R)
    _sep_val_ka!(backend)(R, Pn, Dn, p, d; ndrange = size(R))
    KA.synchronize(backend)
    return R
end

function evaluate_ed!(R::AbstractGPUArray, dR::AbstractGPUArray,
                      basis::SeparableRadial, r::BATCH, ps, st)
    Pn, dPn = evaluate_ed(basis.Pn, r)
    Dn, dDn = _decay_eval_gpu(basis.Dn, r, ps.Dn; withgrad = true)
    p = _device_like(R, [t[1] for t in basis.specidx])
    d = _device_like(R, [t[2] for t in basis.specidx])
    backend = KA.get_backend(R)
    _sep_ed_ka!(backend)(R, dR, Pn, dPn, Dn, dDn, p, d; ndrange = size(R))
    KA.synchronize(backend)
    return R, dR
end

# ---- AtomicOrbitals assembly  ϕ[j,i] = Rnl[j,k]·Ylm[j,y] ----

@kernel function _aorb_val_ka!(Rnlm, @Const(Rn), @Const(Ylm), @Const(kk), @Const(yy))
    j, i = @index(Global, NTuple)
    @inbounds Rnlm[j, i] = Rn[j, kk[i]] * Ylm[j, yy[i]]
end

@kernel function _aorb_ed_ka!(Rnlm, dRnlm, @Const(Rn), @Const(dRn), @Const(Ylm),
                              @Const(dYlm), @Const(X), @Const(r),
                              @Const(kk), @Const(yy))
    j, i = @index(Global, NTuple)
    @inbounds begin
        k = kk[i]; y = yy[i]
        rn = Rn[j, k]; yl = Ylm[j, y]
        Rnlm[j, i] = rn * yl
        drj = X[j] / r[j]
        dRnlm[j, i] = dRn[j, k] * drj * yl + rn * dYlm[j, y]
    end
end

function _gpu_evaluate!(Rnlm, dRnlm, basis::AtomicOrbitals, X, ps, st)
    WITHGRAD = !isnothing(dRnlm)
    r = norm.(X)

    if WITHGRAD
        Rn, dRn = evaluate_ed(basis.Rnl, r, ps.Rnl, st.Rnl)   # SeparableRadial GPU
        Ylm, dYlm = evaluate_ed(basis.Ylm, X)
    else
        Rn = evaluate(basis.Rnl, r, ps.Rnl, st.Rnl)
        Ylm = evaluate(basis.Ylm, X)
    end

    kk = _device_like(Rnlm, [t[1] for t in basis.specidx])
    yy = _device_like(Rnlm, [t[2] for t in basis.specidx])

    backend = KA.get_backend(Rnlm)
    if WITHGRAD
        _aorb_ed_ka!(backend)(Rnlm, dRnlm, Rn, dRn, Ylm, dYlm, X, r,
                              kk, yy; ndrange = size(Rnlm))
    else
        _aorb_val_ka!(backend)(Rnlm, Rn, Ylm, kk, yy; ndrange = size(Rnlm))
    end
    KA.synchronize(backend)
    return nothing
end

function evaluate!(Rnlm::AbstractGPUArray, basis::AtomicOrbitals, X::BATCH, ps, st)
    _gpu_evaluate!(Rnlm, nothing, basis, X, ps, st)
    return Rnlm
end

function evaluate_ed!(Rnlm::AbstractGPUArray, dRnlm::AbstractGPUArray,
                      basis::AtomicOrbitals, X::BATCH, ps, st)
    _gpu_evaluate!(Rnlm, dRnlm, basis, X, ps, st)
    return Rnlm, dRnlm
end
