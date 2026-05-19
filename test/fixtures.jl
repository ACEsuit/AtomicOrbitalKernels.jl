# Shared test fixtures: small `BasisSet`s used across multiple test files.
# Built once on first include; reused thereafter.

using GaussianBasis
using Molecules
using StaticArrays

# Si atom at the origin, Cartesian basis, ACSint backend (pure Julia — no libcint
# binary dependency to fight in CI).
const _SI_ATOM = Molecules.Atom(14, 28.0855, SA[0.0, 0.0, 0.0])

# def2-SVP on Si: 8 shells, max angular momentum l = 2 (s, p, d).
const BS_SI_DEFSVP = BasisSet("def2-SVP", [_SI_ATOM]; spherical=false, lib=:acsint)

# Hydrogen, sto-3g: 1 shell, only s. Smallest possible exercise.
const _H_ATOM = Molecules.Atom(1, 1.008, SA[0.0, 0.0, 0.0])
const BS_H_STO3G = BasisSet("sto-3g", [_H_ATOM]; spherical=false, lib=:acsint)

# Helper: build random (3, B) position matrices on a length scale similar to a
# diatomic molecule (~1.5 Å). Returns both a plain numeric matrix (for
# `Reference.batch_S_pair_ref!`, which is unit-naive) and a Unitful matrix
# carrying `u"angstrom"` (for the public Unitful-only API).
function random_positions(B::Integer; offset = (0.0, 0.0, 0.0), scale = 0.5, rng = Random.default_rng())
    raw = randn(rng, 3, B) .* scale
    raw[1, :] .+= offset[1]
    raw[2, :] .+= offset[2]
    raw[3, :] .+= offset[3]
    return raw, raw .* u"angstrom"
end
