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

# non-trainable "static state", analogous to `_static_params`. Empty fallback;
# bases that carry state (the SpheriCart `Ylm` `Flm`, the radial `poly`, the
# orbital index maps) specialise it.
_static_state(::AbstractP4MLBasis) = NamedTuple()
