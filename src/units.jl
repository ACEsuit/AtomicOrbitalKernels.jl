# Unit handling for position inputs to the batched integral entry points.
#
# Public API is strict: positions must be `AbstractMatrix{<:Unitful.Length}` —
# any length unit is accepted (`u"angstrom"`, `u"nm"`, `u"m"`, …). Plain
# `Real` arrays are deliberately rejected with a clear message. Kernels run on
# bare `Float32`/`Float64`, so units are stripped at the host boundary
# (before any GPU upload).
#
# Implementation note: we never need a "bohr" unit symbol — Unitful ships
# `u"angstrom"` and we already know the Å→Bohr constant. So we convert to Å
# via `uconvert` (works for any `Unitful.Length`) and scale by `ang2bohr`.

const _UNITFUL_HINT = """
positions must carry explicit length units, e.g. `pos .* u"angstrom"` or
`pos .* u"nm"` (any `Unitful.Length` is accepted, including atomic-units
packages' `u"bohr"`). Plain numeric matrices are not accepted — use Unitful
to make the unit choice explicit.
"""

"""
    to_bohr(pos::AbstractMatrix{<:Unitful.Length}, ::Type{FT}=Float64) -> Matrix{FT}

Convert a `(3, B)` matrix of `Unitful.Length` positions to a plain `Matrix{FT}`
holding the same positions in Bohr. Output element type is whatever Float type
the kernel expects (`Float64` on CPU, typically `Float32` on GPU). Strips
units at the host boundary so the kernel sees plain numbers only.
"""
function to_bohr(pos::AbstractMatrix{<:Unitful.Length}, ::Type{FT}=Float64) where {FT}
    a2b = FT(ang2bohr)
    out = Matrix{FT}(undef, size(pos)...)
    @inbounds for j in axes(pos, 2), i in axes(pos, 1)
        out[i, j] = FT(ustrip(uconvert(u"angstrom", pos[i, j]))) * a2b
    end
    return out
end

to_bohr(pos::AbstractMatrix{<:Real}, ::Type{FT}=Float64) where {FT} =
    throw(ArgumentError(_UNITFUL_HINT))
