# learnbasis: fitting GTO radial parameters to 2c overlaps

`learnbasis.jl` tests whether the differentiable 2-center overlap path
(`batch_overlap(orb, XA, XB, ps, st)` + the analytic parameter pullback
`batch_overlap_pb_ps`) can *learn* GTO radial parameters. A reference
STO-3G carbon basis generates overlap matrices for 1000 random C–C pairs
with distances r ∈ (rmin, 4.0)·rvdw (rvdw = 1.7 Å); a copy with ~10%
perturbed (ζ, D) is then fitted by BFGS (Optim.jl) on the LSQ loss, with
ζ log-parameterized and the gradient from a single VJP per iteration.

Run:

```
julia --project=examples/learnbasis examples/learnbasis/learnbasis.jl
```

## Findings

Fit quality (overlap RMSE per matrix entry, 1000 held-out test pairs):

| rmin/rvdw | perturbed | optimized (train) | optimized (test) |
|-----------|-----------|-------------------|------------------|
| 1.6       | 1.6e-3    | 2.3e-7            | 2.3e-7           |
| 0.8       | 4.2e-3    | 1.7e-7            | 1.7e-7           |
| 0.5       | 5.3e-3    | 9.5e-7            | 1.0e-6           |

Parameter recovery (max rel. error of ζ; D compared per radial up to scale):

| radial     | rmin = 1.6 | rmin = 0.8 | rmin = 0.5 |
|------------|-----------|-----------|-----------|
| 1s (core)  | 2.8e-1    | 1.3e-1    | 1.3e-1    |
| 2s         | 7.4e-2    | 5.0e-2    | 1.5e-3    |
| 2p         | 1.2e-1    | 1.0e-2    | 3.8e-3    |

- **The overlaps are always fit well** (3–4 orders of magnitude below the
  perturbed error), and train ≈ test throughout — with 18 parameters and
  25k residuals there is no overfitting.
- **Valence parameters are identifiable from short-range data.** Including
  distances down to 0.85 Å recovers the 2s/2p exponents to ~1e-3 and their
  contraction coefficients to ~1e-4–1e-3.
- **Core parameters are not identifiable from 2c overlaps.** The 1s error
  is unchanged between rmin = 0.8 and 0.5: the tight primitives
  (ζ ≈ 71, 13) decay within ~0.3 Å of the nucleus, so overlaps at any
  chemically meaningful separation carry no information about them. BFGS
  still converges to a genuine stationary point (|g| ≲ 1e-10) — just a
  different point on the near-flat core manifold. Pinning down the core
  would need observables that weight the near-nucleus region (kinetic,
  nuclear attraction).
- **D is identifiable only up to a per-radial scale** because the
  contracted radials are normalized; for K = 1 contractions ∂/∂D ≡ 0
  exactly (which is why the example uses STO-3G, where all K = 3).
