# Deterministic example atomic-orbital bases. These are placeholders for testing
# and demos; real (data-driven) constructors arrive with the basis-set sourcing
# work. TODO: revisit once `.gbs`-style data loading lands.

# build an example GSRadials of the requested type: for each l in 0:N1-1, the
# N1*N2 pairs (n1, n2) set the radial quantum number (n1) and a deterministic
# exponent/coefficient set; `K` columns give a contraction. The radial power is
# derived (GTO: 0; STO: n1-1), not stored. With `length(zlist)` species the (ζ,D)
# gain a species axis: σ=1 reproduces the single-species basis exactly, further
# species are mild deterministic perturbations so their slices are distinct.
function _example_radial(::Type{TR}, N1, N2; K::Int = 1, T::Type = Float64,
                         zlist = (1,)) where {TR <: GSRadials}
    spec   = NT_NL[]
    nnspec = NT_NNL[]
    ζrows  = Vector{T}[]
    Drows  = Vector{T}[]
    for l = 0:N1-1
        n = 0
        for n1 = 1:N1, n2 = 1:N2
            n += 1
            push!(spec,   (n = n, l = l))
            push!(nnspec, (n1 = n1, n2 = n2, l = l))
            push!(ζrows,  T[ T(0.5) * (n2 + j) + l for j = 1:K ])
            push!(Drows,  TR === GaussianTypeRadials ? ones(T, K) : T[ 1 / j for j = 1:K ])
        end
    end
    ζ2 = permutedims(reduce(hcat, ζrows))   # [nRad × K]
    D2 = permutedims(reduce(hcat, Drows))
    nRad = size(ζ2, 1)
    NZ = length(zlist)
    ζ = zeros(T, nRad, K, NZ)               # [nRad × K × NZ]
    D = zeros(T, nRad, K, NZ)
    for σ = 1:NZ
        ζ[:, :, σ] .= ζ2 .+ T(0.1) * (σ - 1)
        D[:, :, σ] .= D2 .* (1 + T(0.05) * (σ - 1))
    end
    return TR(ζ, D, spec, nnspec, zlist)
end

"""
    gaussian_orbitals(N1=4, N2=3; length_unit, K=1, T=Float64, nspecies=1, zlist=1:nspecies)

A deterministic example Gaussian-type atomic-orbital basis. With `nspecies>1` the
radial `(ζ,D)` are species-indexed (labels `zlist`); species 1 reproduces the
single-species basis. `length_unit` is required (see `AtomicOrbitals`).
"""
gaussian_orbitals(N1 = 4, N2 = 3; length_unit, K::Int = 1, T = Float64,
                  nspecies::Int = 1, zlist = ntuple(i -> i, nspecies)) =
        AtomicOrbitals(_example_radial(GaussianTypeRadials, N1, N2;
                                       K = K, T = T, zlist = zlist),
                       _default_ylm(N1 - 1); length_unit = length_unit)

"""
    slater_orbitals(N1=4, N2=3; length_unit, K=1, T=Float64, nspecies=1, zlist=1:nspecies)

A deterministic example Slater-type atomic-orbital basis (`K>1` gives a
contracted radial). With `nspecies>1` the radial `(ζ,D)` are species-indexed.
`length_unit` is required (see `AtomicOrbitals`).
"""
slater_orbitals(N1 = 4, N2 = 3; length_unit, K::Int = 1, T = Float64,
                nspecies::Int = 1, zlist = ntuple(i -> i, nspecies)) =
        AtomicOrbitals(_example_radial(SlaterTypeRadials, N1, N2;
                                       K = K, T = T, zlist = zlist),
                       _default_ylm(N1 - 1); length_unit = length_unit)
