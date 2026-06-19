# DataPlan.md — Sourcing atomic-orbital basis-set data (Workstream C)

Plan for pulling real Gaussian basis-set data into AtomicOrbitalKernels: bundle a
few common sets in the repo, download the rest on demand from the Basis Set
Exchange (BSE), and cache downloads locally so they aren't re-fetched.

---

## 1. Goal & context
Today the learnable radials (`GaussianTypeRadials` ζ/D) are built by **toy**
constructors (`src/orbitals/utils.jl`: *"real data-driven constructors arrive
with the basis-set sourcing work"*), and named GTO sets reach the integral path
only via `GaussianBasis.BasisSet("def2-SVP", …)` (`compile_basis(::BasisSet)` in
`src/compiled_basis.jl`). This plan adds a data source that feeds **both**:
- the eval side — real ζ/D for `GaussianTypeRadials` / `AtomicOrbitals`;
- the integral side — `compile_basis` from parsed shells (eventually replacing
  the hard `GaussianBasis` dependency; the code already notes moving
  `compile_basis(::BasisSet)` into a `GaussianBasisExt`).

## 2. Is BSE easy to use? Yes
- **REST API**, no key:
  `http://www.basissetexchange.org/api/basis/{name}/format/{fmt}/?elements=1,6,8&version=N`
  - e.g. `…/basis/def2-svp/format/json/?elements=14` (Si)
  - e.g. `…/basis/sto-3g/format/json/?elements=1,6,7,8`
- **Format: JSON** (recommended). Native BSE schema is trivial to parse:
  `elements[Z].electron_shells[] = { angular_momentum[], exponents[],
  coefficients[[]], function_type (gto / gto_spherical) }`. Lossless, carries
  spherical/cartesian and references. (gaussian94/`.gbs` is the alternative,
  GaussianBasis-readable, but JSON is easier and richer.)
- **Listing / metadata**: `/api/metadata` (all names + elements covered),
  `/api/references/{name}/format/bib`.
- **Licensing: CC-BY 4.0 — redistributable.** We may bundle basis files in the
  repo with attribution: cite Pritchard et al., *J. Chem. Inf. Model.* 2019,
  59(11), 4814 (doi:10.1021/acs.jcim.9b00725), and carry each basis's own
  references (present in the JSON). BSE code is BSD-3.
- **Etiquette**: beta API, no hard rate limit; set a `User-Agent`, and cache (we
  do). Full data also mirrored at github.com/MolSSI-BSE/basis_set_exchange
  (`data/`) and a Zenodo snapshot — usable if we ever want a vendored bundle.
- **Note**: `GaussianBasis.jl` already bundles BSE data and loads named sets
  offline (`lib=:acsint`, pure Julia). So BSE-direct is about (a) dropping that
  dependency, (b) sets it doesn't bundle, and (c) feeding the eval/learnable side
  with raw exponents/coefficients.

## 3. Julia mechanism: bundle + download + cache
Use **Scratch.jl + Downloads (stdlib) + Preferences.jl** — the standard combo.
(DataDeps.jl and Pkg Artifacts are the wrong fit: both want a *fixed,
pre-registered* set, not "download any basis by name".)
- **Bundle**: ~5 common sets as JSON under `data/` (sto-3g, 6-31G*, def2-SVP,
  cc-pVDZ, cc-pVTZ) → always available offline, no network in CI.
- **Fetch on demand**: `Downloads.download(url, tmp)` from the BSE API.
- **Cache**: a Scratch space keyed by name+elements+version,
  `~/.julia/scratchspaces/<pkg-uuid>/bse/<name>_<elts>.json`; persists across
  sessions, GC'd when the package is removed. Lookup order: bundled → cache →
  download.
- **Config** via Preferences: `offline` (bundled+cache only, error otherwise —
  for reproducibility/CI), `cache_dir`, and a `bse_url` mirror override.
- **Integrity**: store/verify a SHA-256 of each cached file (SHA is stdlib).

## 4. Architecture
A small `src/basis_data.jl` (or a `BasisData` submodule):
```
load_basis(name, elements; offline=…) :: ParsedBasis     # bundled → cache → download → parse JSON
ParsedBasis: per-element shells (l, exponents, coefficients, spherical::Bool, refs)
  → compile_basis(::ParsedBasis)                          # integral side, new method alongside
                                                          #   the existing compile_basis(::BasisSet)
  → GaussianTypeRadials / AtomicOrbitals constructors     # eval side: real ζ, D from the shells
```
A neutral `ParsedBasis` decouples the source from both consumers, so neither the
GaussianBasis path nor the kernels change initially.

## 5. Dependencies to add (needs sign-off)
`Scratch`, `Preferences`, and a JSON parser (`JSON3` — fast — or `JSON`).
`Downloads` and `SHA` are stdlib. All lightweight.

## 6. Phasing
1. **Eval-side first** (closes the placeholder-constructor gap, touches nothing
   else): `load_basis` + bundled sets + Scratch cache + a
   `gaussian_orbitals_from_basis(name, elements)`-style constructor mapping
   contracted shells → `GaussianTypeRadials` (ζ, D, poly) + Ylm.
2. **Integral-side**: `compile_basis(::ParsedBasis)` from the same parsed shells.
3. **GaussianBasis demotion**: move `compile_basis(::BasisSet)` to a
   `GaussianBasisExt` extension; BSE-JSON becomes the primary named-set path.

## 7. Tests / CI
- Offline: load a **bundled** set, assert parsed shells match a known reference
  (exponents/coefficients for e.g. STO-3G H). Runs in CI with no network.
- Live download: an **opt-in, env-gated** test (like the GPU tests) that fetches
  one small set from BSE and checks the cache is populated — never run in normal
  CI so we don't hit BSE.
- Round-trip: `load_basis → GaussianTypeRadials → evaluate` produces finite,
  correctly-shaped output; optionally cross-check values against the
  GaussianBasis path for the same named set.

## 8. Open decisions
- JSON (recommended) vs gaussian94 storage format.
- Which sets to bundle (proposed: sto-3g, 6-31G*, def2-SVP, cc-pVDZ, cc-pVTZ).
- Dep choice: `JSON3` vs `JSON`.
- Scope of first PR: eval-side only (recommended) vs eval+integral together.

## References
- BSE REST API: https://molssi-bse.github.io/basis_set_exchange/ (web_api / user_api)
- Data repo: https://github.com/MolSSI-BSE/basis_set_exchange (`data/`)
- License: CC-BY 4.0; cite Pritchard et al. 2019, doi:10.1021/acs.jcim.9b00725
- Scratch.jl, Preferences.jl, Downloads (stdlib), SHA (stdlib)
