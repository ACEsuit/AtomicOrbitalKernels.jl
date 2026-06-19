# Deterministic example atomic-orbital bases. These are placeholders for testing
# and demos; real (data-driven) constructors arrive with the basis-set sourcing
# work. TODO: revisit once `.gbs`-style data loading lands.

# build an example GSRadials of the requested type: for each l in 0:N1-1, the
# N1*N2 pairs (n1, n2) give the polynomial degree (n1-1) and a deterministic
# exponent/coefficient set; `K` columns give a contraction.
function _example_radial(::Type{TR}, N1, N2; K::Int = 1, T::Type = Float64) where {TR <: GSRadials}
    spec   = NT_NL[]
    nnspec = NT_NNL[]
    poly   = Int[]
    ζrows  = Vector{T}[]
    Drows  = Vector{T}[]
    for l = 0:N1-1
        n = 0
        for n1 = 1:N1, n2 = 1:N2
            n += 1
            push!(spec,   (n = n, l = l))
            push!(nnspec, (n1 = n1, n2 = n2, l = l))
            push!(poly,   n1 - 1)
            push!(ζrows,  T[ T(0.5) * (n2 + j) + l for j = 1:K ])
            push!(Drows,  TR === GaussianTypeRadials ? ones(T, K) : T[ 1 / j for j = 1:K ])
        end
    end
    ζ = permutedims(reduce(hcat, ζrows))   # [nRad × K]
    D = permutedims(reduce(hcat, Drows))
    return TR(ζ, D, poly, spec, nnspec)
end

"""
    gaussian_orbitals(N1=4, N2=3; K=1, T=Float64)

A deterministic example Gaussian-type atomic-orbital basis.
"""
gaussian_orbitals(N1 = 4, N2 = 3; K::Int = 1, T = Float64) =
        AtomicOrbitals(_example_radial(GaussianTypeRadials, N1, N2; K = K, T = T),
                       _default_ylm(N1 - 1))

"""
    slater_orbitals(N1=4, N2=3; K=1, T=Float64)

A deterministic example Slater-type atomic-orbital basis (`K>1` gives a
contracted radial).
"""
slater_orbitals(N1 = 4, N2 = 3; K::Int = 1, T = Float64) =
        AtomicOrbitals(_example_radial(SlaterTypeRadials, N1, N2; K = K, T = T),
                       _default_ylm(N1 - 1))
