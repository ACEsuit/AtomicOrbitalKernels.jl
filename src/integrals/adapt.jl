"""
    adapt_basis(basis::CompiledBasis, ArrayCtor, ::Type{FT}=eltype(basis.coef)) -> CompiledBasis

Move a `CompiledBasis` to a different backend / element type. `ArrayCtor` is
the device array constructor (`Array`, `CuArray`, `MtlArray`, `ROCArray`, …)
and `FT` is the float type the kernel will run at.

```julia
b_gpu = adapt_basis(b, CuArray, Float32)   # CUDA, Float32 kernel
b_mtl = adapt_basis(b, MtlArray, Float32)  # Metal (Float32 is mandatory)
```
"""
function adapt_basis(basis::CompiledBasis, ArrayCtor, ::Type{FT}=eltype(basis.coef)) where {FT}
    ls           = ArrayCtor(basis.ls)
    nprim        = ArrayCtor(basis.nprim)
    prim_offset  = ArrayCtor(basis.prim_offset)
    coef         = ArrayCtor(FT.(basis.coef))
    α            = ArrayCtor(FT.(basis.α))
    nbf          = ArrayCtor(basis.nbf)
    basis_offset = ArrayCtor(basis.basis_offset)
    return CompiledBasis{FT,typeof(ls),typeof(coef)}(
        basis.Lmax, basis.nshells, ls, nprim, prim_offset, coef, α,
        nbf, basis_offset, basis.nbf_total
    )
end
