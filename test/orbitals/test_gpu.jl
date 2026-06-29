# Device tests for AtomicOrbitals evaluation. The backend (`dev` / `gpu_backend`)
# comes from utils_gpu.jl; with no functional GPU, `dev = identity` and these
# run on the CPU backend (in Float32).

using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed
import AtomicOrbitalKernels as AOK
using StaticArrays, LuxCore
using LinearAlgebra, Random, Test 
rng = MersenneTwister(1234)
include(joinpath(@__DIR__, "..", "utils_gpu.jl"))

@testset "AtomicOrbitals eval ($(gpu_backend), Float32)" begin
    basis = gaussian_orbitals(; length_unit = :bohr)
    Xh = [@SVector randn(3) for _ = 1:64]
    Pc = AOK.evaluate_ref(basis, Xh)              # CPU forward reference (Float64)
    _, dPc = evaluate_ed(basis, Xh)               # CPU KA gradient (Float64)

    # params + state moved to the device (indices/poly/Flm live in the state);
    # `_static_params` returns the ζ/D as plain `Array`s (the basis stores MArrays)
    ps = LuxCore.initialparameters(rng, basis)
    _st = LuxCore.initialstates(rng, basis)
    st = (Rnl = _st.Rnl, Ylm = (Flm = Float32.(_st.Ylm.Flm),), 
            iR = _st.iR, iY = _st.iY)
    psg = dev(ps)
    stg = dev(st)
    Xg = dev([SVector{3, Float32}(x) for x in Xh])

    Pg = evaluate(basis, Xg, psg, stg)
    @test dev === identity || !(Pg isa Array)    # output stays on the device
    @test norm(Array(Pg) .- Float32.(Pc)) / norm(Pc) < 1f-3

    _, dPg = evaluate_ed(basis, Xg, psg, stg)
    dPh = Array(dPg)
    num = maximum(norm(dPh[i] - SVector{3, Float32}(dPc[i])) for i in eachindex(dPh))
    den = maximum(norm(SVector{3, Float32}(v)) for v in dPc)
    @test num / den < 1f-2

    # parameter pullback + rrule on the device, vs the CPU reference
    psc = AOK._static_params(basis); stc = AOK._static_state(basis)
    ∂P = randn(length(Xh), length(basis))
    _, pbc = AOK.rrule(evaluate, basis, Xh, psc, stc)
    _, _, ∂Xc, ∂ps_c, _ = pbc(∂P)

    ∂Pg = dev(Float32.(∂P))
    _, pbg = AOK.rrule(evaluate, basis, Xg, psg, stg)
    _, _, ∂Xg, ∂ps_g, _ = pbg(∂Pg)

    @test dev === identity || !(∂ps_g.Rnl.ζ isa Array)   # gradients stay on device
    @test norm(Array(∂ps_g.Rnl.ζ) .- Float32.(∂ps_c.Rnl.ζ)) / norm(∂ps_c.Rnl.ζ) < 1f-2
    @test norm(Array(∂ps_g.Rnl.D) .- Float32.(∂ps_c.Rnl.D)) / norm(∂ps_c.Rnl.D) < 1f-2
    ∂Xgh = Array(∂Xg)
    nX = maximum(norm(∂Xgh[i] - SVector{3,Float32}(∂Xc[i])) for i in eachindex(∂Xc))
    dX = maximum(norm(SVector{3,Float32}(v)) for v in ∂Xc)
    @test nX / dX < 1f-2
end

# multi-species: exercise the species-indexed radial + species-scatter pullback
# kernels on the device, via the internal (positions, sidx) layer.
@testset "AtomicOrbitals multi-species ($(gpu_backend), Float32)" begin
    basis = gaussian_orbitals(4, 3; length_unit = :bohr, nspecies = 2, zlist = (1, 6))
    nX = 64
    Xh = [@SVector randn(3) for _ = 1:nX]
    sidx = rand(1:2, nX)
    psc = AOK._static_params(basis); stc = AOK._static_state(basis)

    # CPU references (Float64) through the internal (positions, sidx) layer
    Pc = evaluate(basis, Xh, sidx, psc, stc)
    _, dPc = evaluate_ed(basis, Xh, sidx, psc, stc)
    ∂P = randn(nX, length(basis))
    ∂psc = AOK.pullback_ps(∂P, basis, Xh, sidx, psc, stc)

    Xg = dev([SVector{3, Float32}(x) for x in Xh])
    sg = dev(sidx)

    ps = LuxCore.initialparameters(rng, basis)
    _st = LuxCore.initialstates(rng, basis)
    st = (Rnl = _st.Rnl, Ylm = (Flm = Float32.(_st.Ylm.Flm),), 
            iR = _st.iR, iY = _st.iY)
    psg = dev(ps)
    stg = dev(st)

    Pg = evaluate(basis, Xg, sg, psg, stg)
    @test dev === identity || !(Pg isa Array)
    @test norm(Array(Pg) .- Float32.(Pc)) / norm(Pc) < 1f-3

    _, dPg = evaluate_ed(basis, Xg, sg, psg, stg)
    dPh = Array(dPg)
    num = maximum(norm(dPh[i] - SVector{3, Float32}(dPc[i])) for i in eachindex(dPh))
    den = maximum(norm(SVector{3, Float32}(v)) for v in dPc)
    @test num / den < 1f-2

    ∂Pg = dev(Float32.(∂P))
    ∂psg = AOK.pullback_ps(∂Pg, basis, Xg, sg, psg, stg)
    @test dev === identity || !(∂psg.Rnl.ζ isa Array)
    @test norm(Array(∂psg.Rnl.ζ) .- Float32.(∂psc.Rnl.ζ)) / norm(∂psc.Rnl.ζ) < 1f-2
    @test norm(Array(∂psg.Rnl.D) .- Float32.(∂psc.Rnl.D)) / norm(∂psc.Rnl.D) < 1f-2
end
