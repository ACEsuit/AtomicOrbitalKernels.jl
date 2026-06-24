using BenchmarkTools
using AtomicOrbitalKernels
using AtomicOrbitalKernels: evaluate, evaluate_ed, pullback_ps, nspecies, _aorb_rrule,
                            _powers
using StaticArrays, LinearAlgebra, Random

SUITE = BenchmarkGroup()

# Register the forward evals and the pullbacks for one Gaussian-orbital basis on a
# given backend. `dev` moves data to the device (identity on CPU); evaluate /
# pullback synchronise internally, so they time correctly on the GPU too. Inputs
# go through the internal (positions, `sidx`) layer with a random species index
# per point, so the species-indexed radial + species-scatter pullback are timed.
# `ζ,D` are extracted as plain `Array`s (the basis stores MArrays). Kernels of
# interest: `_gtostoradials_pb_ka!` (radial param pullback), `_aorb_pbrad_ka!`
# (orbital→radial scatter), `_aorb_pbx_ka!` (X-pullback).
function _add_orbital_group!(group, dev, T, basis, nX)
    Xh = [randn(SVector{3, T}) for _ = 1:nX]
    X  = dev(Xh)
    r  = dev(T.(norm.(Xh)))
    NZ = nspecies(basis.Rnl)
    sidx = dev(rand(1:NZ, nX))
    ps = (Rnl = (ζ = dev(T.(Array(basis.Rnl.ζ))), D = dev(T.(Array(basis.Rnl.D)))),
          Ylm = NamedTuple())
    st = (Rnl = (poly = dev(collect(_powers(basis.Rnl))),),
          Ylm = (Flm = dev(T.(basis.Ylm.Flm)),),
          iR = dev(collect(basis.radidx)), iY = dev(collect(basis.ylmidx)))
    ∂P = dev(randn(T, nX, length(basis)))
    ∂R = dev(randn(T, nX, length(basis.Rnl)))

    group["evaluate"]        = @benchmarkable evaluate($basis, $X, $sidx, $ps, $st)
    group["evaluate_ed"]     = @benchmarkable evaluate_ed($basis, $X, $sidx, $ps, $st)
    # radial parameter pullback in isolation (kernel `_gtostoradials_pb_ka!`)
    group["radial_pullback"] = @benchmarkable pullback_ps($∂R, $(basis.Rnl), $r,
                                                          $sidx, $(ps.Rnl), $(st.Rnl))
    # orbital parameter pullback (kernel `_aorb_pbrad_ka!` + radial)
    group["pullback_ps"]     = @benchmarkable pullback_ps($∂P, $basis, $X, $sidx, $ps, $st)
    # rrule pullback: X-pullback (kernel `_aorb_pbx_ka!`) + parameter pullback
    _, pb = _aorb_rrule(basis, X, X, sidx, ps, st)
    group["rrule_pullback"]  = @benchmarkable $pb($∂P)
    return group
end

const _NX = 2000
# K=1 (segmented) and K=8 (contracted) radials; nRad≈48, norb≈192. `-Z4` variants
# carry 4 species, exercising the species-scatter parameter pullback.
_bases() = ("K1"    => gaussian_orbitals(4, 3; K = 1),
            "K8"    => gaussian_orbitals(4, 3; K = 8),
            "K1-Z4" => gaussian_orbitals(4, 3; K = 1, nspecies = 4),
            "K8-Z4" => gaussian_orbitals(4, 3; K = 8, nspecies = 4))

function _add_device!(suite, key, dev, T)
    suite[key] = BenchmarkGroup()
    for (label, basis) in _bases()
        suite[key][label] = BenchmarkGroup()
        _add_orbital_group!(suite[key][label], dev, T, basis, _NX)
    end
end

_add_device!(SUITE, "cpu", identity, Float64)

# benchmark the CUDA backend too when a functional GPU is present
try
    using CUDA
    if CUDA.functional()
        _add_device!(SUITE, "cuda", CUDA.cu, Float32)
    else
        @info "benchmarks: CUDA present but not functional — CPU only"
    end
catch
    @info "benchmarks: CUDA not available — CPU only"
end
