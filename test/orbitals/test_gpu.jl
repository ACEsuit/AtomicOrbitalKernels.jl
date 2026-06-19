# Opt-in GPU tests for AtomicOrbitals evaluation. The GPU backend is detected in
# runtests.jl (`_gpu` / `_gpu_name`); skipped when no functional backend exists.

using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed
import AtomicOrbitalKernels as AOK
using StaticArrays
using LinearAlgebra

if _gpu !== nothing
    @testset "GPU AtomicOrbitals eval ($(_gpu_name), Float32)" begin
        basis = gaussian_orbitals()
        Xh = [@SVector randn(3) for _ = 1:64]
        Pc = AOK.evaluate_ref(basis, Xh)              # CPU forward reference (Float64)
        _, dPc = evaluate_ed(basis, Xh)               # CPU KA gradient (Float64)

        # params + state moved to the device (indices/poly/Flm live in the state)
        Xg = _gpu([SVector{3, Float32}(x) for x in Xh])
        psg = (Rnl = (ζ = _gpu(Float32.(basis.Rnl.ζ)), D = _gpu(Float32.(basis.Rnl.D))),
               Ylm = NamedTuple())
        stg = (Rnl = (poly = _gpu(collect(basis.Rnl.poly)),),
               Ylm = (Flm = _gpu(basis.Ylm.Flm),),
               iR = _gpu(collect(basis.radidx)),
               iY = _gpu(collect(basis.ylmidx)))

        Pg = evaluate(basis, Xg, psg, stg)
        @test !(Pg isa Array)                        # ran on the device
        @test norm(Array(Pg) .- Float32.(Pc)) / norm(Pc) < 1f-3

        _, dPg = evaluate_ed(basis, Xg, psg, stg)
        dPh = Array(dPg)
        num = maximum(norm(dPh[i] - SVector{3, Float32}(dPc[i])) for i in eachindex(dPh))
        den = maximum(norm(SVector{3, Float32}(v)) for v in dPc)
        @test num / den < 1f-2
    end
end
