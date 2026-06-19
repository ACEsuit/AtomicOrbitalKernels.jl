# package-global helpers

# invert a vector of "natural indices" into a value -> position map
_invmap(a::AbstractVector) = Dict{eltype(a), Int}(a[i] => i for i = 1:length(a))

# copy a host index/structural vector onto the backend of `ref`
_device_like(ref, v::AbstractVector) =
        copyto!(similar(ref, eltype(v), length(v)), v)

# allocate value / gradient output arrays on the backend of `X`; the KA kernels
# fill every entry, so the arrays are left uninitialised.
_alloc_val(basis, X, ps, st) =
        similar(X, _valtype(basis, X, ps, st), length(X), length(basis))
_alloc_grad(basis, X, ps, st) =
        similar(X, _gradtype(basis, X, ps, st), length(X), length(basis))

# `_static_params` (params used by the parameter-free evaluation) and
# `_static_state` (non-trainable state) are both owned by this package — neither
# is imported from Polynomials4ML — so their empty fallbacks are defined on
# `Any`. Bases that carry params/state (the radial ζ/D/poly, the SpheriCart
# `Ylm` `Flm`, the orbital index maps) specialise them.
_static_params(::Any) = NamedTuple()
_static_state(::Any) = NamedTuple()
