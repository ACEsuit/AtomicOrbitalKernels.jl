# McMurchie–Davidson Hermite-to-Cartesian E-coefficient recursion (Helgaker,
# Jørgensen, Olsen, eqs. 9.2.11–9.2.15 and 9.5.6–9.5.7). Pedagogical scalar
# implementation used as the correctness oracle for the KA kernels.

function alloc_E(B1, B2)
    am1 = B1.l
    am2 = B2.l
    similar(B1.atom.xyz, 3, am1 + am2 + 2, am1 + 1, am2 + 1)
end

function alloc_E3(B1, B2, B3)
    am1 = B1.l
    am2 = B2.l
    am3 = B3.l
    similar(B1.atom.xyz, 3, am1 + am2 + am3 + 3, am1 + 1, am2 + 1, am3 + 1)
end

@inline function generate_E_matrix!(E, maxam1, maxam2, P, A, B, a, b)
    E .= 0

    p = a + b
    AB = A - B
    PA = P - A
    PB = P - B
    µ = a * b / p
    @inbounds @. E[:, 1, 1, 1] = exp(-µ * AB^2)
    oo2p = 1 / (2 * p)
    @inbounds for j = 1:maxam2+1
        for i = 1:maxam1+1
            if i > 1
                @views E[:, 1, i, j] .+= PA .* E[:, 1, i-1, j] .+ E[:, 2, i-1, j]
            elseif j > 1
                @views E[:, 1, i, j] .+= PB .* E[:, 1, i, j-1] .+ E[:, 2, i, j-1]
            end
            for t = 2:i+j-1
                if i > 1
                    @. @views E[:, t, i, j] += PA * E[:, t, i-1, j] + t * E[:, t+1, i-1, j] + oo2p * E[:, t-1, i-1, j]
                elseif j > 1
                    @. @views E[:, t, i, j] += PB * E[:, t, i, j-1] + t * E[:, t+1, i, j-1] + oo2p * E[:, t-1, i, j-1]
                end
            end
        end
    end
end

@inline function generate_E3_matrix!(E3, maxam1, maxam2, maxam3,
                                     P, Q, A, B, C,
                                     a::Real, b::Real, c::Real)
    fill!(E3, zero(eltype(E3)))
    p = a + b
    q = p + c

    AB = A - B
    PC = P - C
    QA = Q - A
    QB = Q - B
    QC = Q - C

    µ = a * b / p
    ν = p * c / q

    @inbounds @. E3[:, 1, 1, 1, 1] = exp(-µ * AB^2 - ν * PC^2)
    oo2q = 1 / (2 * q)

    @inbounds for k = 1:maxam3+1, j = 1:maxam2+1, i = 1:maxam1+1
        if i > 1
            @views @. E3[:, 1, i, j, k] += QA * E3[:, 1, i-1, j, k] + E3[:, 2, i-1, j, k]
        elseif j > 1
            @views @. E3[:, 1, i, j, k] += QB * E3[:, 1, i, j-1, k] + E3[:, 2, i, j-1, k]
        elseif k > 1
            @views @. E3[:, 1, i, j, k] += QC * E3[:, 1, i, j, k-1] + E3[:, 2, i, j, k-1]
        end
        for t = 2:i+j+k-1
            if i > 1
                @views @. E3[:, t, i, j, k] += QA * E3[:, t, i-1, j, k] + t * E3[:, t+1, i-1, j, k] + oo2q * E3[:, t-1, i-1, j, k]
            elseif j > 1
                @views @. E3[:, t, i, j, k] += QB * E3[:, t, i, j-1, k] + t * E3[:, t+1, i, j-1, k] + oo2q * E3[:, t-1, i, j-1, k]
            elseif k > 1
                @views @. E3[:, t, i, j, k] += QC * E3[:, t, i, j, k-1] + t * E3[:, t+1, i, j, k-1] + oo2q * E3[:, t-1, i, j, k-1]
            end
        end
    end
end
