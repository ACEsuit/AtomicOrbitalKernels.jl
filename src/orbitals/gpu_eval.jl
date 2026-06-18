# GPU-first KernelAbstractions evaluation for `AtomicOrbitals`.
#
# The composite ϕ = Pn·Dn·Ylm is evaluated by computing the three factor bases
# into device matrices and assembling the product (and its spatial gradient) in
# a KA kernel. `Pn` (MonoBasis) and `Ylm` (SpheriCart) already have GPU
# evaluation paths; the radial decay `Dn` and the product assembly are the
# kernels below. The radial parameters are taken from the Lux `ps.Dn = (ζ, D)`,
# which on the GPU live on the device. These kernels run on any KA backend, so
# the same path also works on the CPU backend.

# ---- radial decay  Dn(r) = Σ_m D[n,m] exp(-ζ[n,m] f(r)) ----

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

function _radial_eval_gpu(basis::RadialDecay, x, ps; withgrad::Bool)
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

# ---- product assembly  Rnl[j,i] = Pn[j,b1]·Dn[j,b2]·Ylm[j,b3] ----

@kernel function _aorb_val_ka!(Rnl, @Const(Pn), @Const(Dn), @Const(Ylm),
                               @Const(b1), @Const(b2), @Const(b3))
    j, i = @index(Global, NTuple)
    @inbounds Rnl[j, i] = Pn[j, b1[i]] * Dn[j, b2[i]] * Ylm[j, b3[i]]
end

@kernel function _aorb_ed_ka!(Rnl, dRnl, @Const(Pn), @Const(Dn), @Const(Ylm),
                              @Const(dPn), @Const(dDn), @Const(dYlm),
                              @Const(X), @Const(R),
                              @Const(b1), @Const(b2), @Const(b3))
    j, i = @index(Global, NTuple)
    @inbounds begin
        i1 = b1[i]; i2 = b2[i]; i3 = b3[i]
        p = Pn[j, i1]; d = Dn[j, i2]; y = Ylm[j, i3]
        Rnl[j, i] = p * d * y
        drj = X[j] / R[j]
        dRnl[j, i] = dPn[j, i1] * drj * d * y +
                     p * dDn[j, i2] * drj * y +
                     p * d * dYlm[j, i3]
    end
end

# copy a host index vector to a device vector matching `ref`'s backend
_device_like(ref, v::AbstractVector) =
        copyto!(similar(ref, eltype(v), length(v)), v)

# ---- GPU evaluation entry points (Lux params on device) ----

function _gpu_evaluate!(Rnl, dRnl, basis::AtomicOrbitals, X, ps, st)
    WITHGRAD = !isnothing(dRnl)
    nX = length(X)
    nB = length(basis)
    R = norm.(X)

    # Pn (MonoBasis) and Ylm (SpheriCart) are parameter-free with GPU paths.
    if WITHGRAD
        Pn, dPn = evaluate_ed(basis.Pn, R)
        Dn, dDn = _radial_eval_gpu(basis.Dn, R, ps.Dn; withgrad = true)
        Ylm, dYlm = evaluate_ed(basis.Ylm, X)
    else
        Pn = evaluate(basis.Pn, R)
        Dn = _radial_eval_gpu(basis.Dn, R, ps.Dn; withgrad = false)
        Ylm = evaluate(basis.Ylm, X)
    end

    b1 = _device_like(Rnl, [t[1] for t in basis.specidx])
    b2 = _device_like(Rnl, [t[2] for t in basis.specidx])
    b3 = _device_like(Rnl, [t[3] for t in basis.specidx])

    backend = KA.get_backend(Rnl)
    if WITHGRAD
        _aorb_ed_ka!(backend)(Rnl, dRnl, Pn, Dn, Ylm, dPn, dDn, dYlm, X, R,
                              b1, b2, b3; ndrange = (nX, nB))
    else
        _aorb_val_ka!(backend)(Rnl, Pn, Dn, Ylm, b1, b2, b3; ndrange = (nX, nB))
    end
    KA.synchronize(backend)
    return nothing
end

function evaluate!(Rnl::AbstractGPUArray, basis::AtomicOrbitals, X::BATCH, ps, st)
    _gpu_evaluate!(Rnl, nothing, basis, X, ps, st)
    return Rnl
end

function evaluate_ed!(Rnl::AbstractGPUArray, dRnl::AbstractGPUArray,
                      basis::AtomicOrbitals, X::BATCH, ps, st)
    _gpu_evaluate!(Rnl, dRnl, basis, X, ps, st)
    return Rnl, dRnl
end
