# Device tests for AtomicOrbitals evaluation. The backend (`dev` / `gpu_backend`)
# comes from utils_gpu.jl; with no functional GPU, `dev = identity` and these
# run on the CPU backend (in Float32).

using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed
import AtomicOrbitalKernels as AOK
using StaticArrays
using LinearAlgebra
include(joinpath(@__DIR__, "..", "utils_gpu.jl"))

@testset "AtomicOrbitals eval ($(gpu_backend), Float32)" begin
    basis = gaussian_orbitals()
    Xh = [@SVector randn(3) for _ = 1:64]
    Pc = AOK.evaluate_ref(basis, Xh)              # CPU forward reference (Float64)
    _, dPc = evaluate_ed(basis, Xh)               # CPU KA gradient (Float64)

    # params + state moved to the device (indices/poly/Flm live in the state)
    Xg = dev([SVector{3, Float32}(x) for x in Xh])
    psg = (Rnl = (ζ = dev(Float32.(basis.Rnl.ζ)), D = dev(Float32.(basis.Rnl.D))),
           Ylm = NamedTuple())
    stg = (Rnl = (poly = dev(collect(basis.Rnl.poly)),),
           Ylm = (Flm = dev(basis.Ylm.Flm),),
           iR = dev(collect(basis.radidx)),
           iY = dev(collect(basis.ylmidx)))

    Pg = evaluate(basis, Xg, psg, stg)
    @test dev === identity || !(Pg isa Array)    # output stays on the device
    @test norm(Array(Pg) .- Float32.(Pc)) / norm(Pc) < 1f-3

    _, dPg = evaluate_ed(basis, Xg, psg, stg)
    dPh = Array(dPg)
    num = maximum(norm(dPh[i] - SVector{3, Float32}(dPc[i])) for i in eachindex(dPh))
    den = maximum(norm(SVector{3, Float32}(v)) for v in dPc)
    @test num / den < 1f-2
end
