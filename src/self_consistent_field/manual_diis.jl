"""
Standard fixed depth DIIS as described e.g. in [https://doi.org/10.1051/m2an/2021069]
(Its a review, see Pulay for the introduction of DIIS I think)
"""
struct DIIS
    m::Int                            # maximal history size
    iterates::Vector{Any}             # xₙ
    residuals::Vector{Any}            # rₙ
end
DIIS(;m=10) = DIIS(m, [], [])
depth(diis::DIIS) = length(diis.iterates)

# Actualize diis lists and remove old iterates if needed
function Base.push!(diis::DIIS, xₙ, Rₙ)
    if depth(diis) + 1 > diis.m
        norms = norm.(M.residuals)
        _, idx = findmax(norms)
        deleteat!(diis.iterates)
        deleteat!(diis.residuals)
    end
    push!(diis.iterates,  deepcopy(vec(xₙ)))
    push!(diis.residuals, deepcopy(vec(Rₙ)))
    @assert depth(diis) <= diis.m + 1
    @assert length(diis.residuals) == depth(diis)
end

function (diis::DIIS)(Pd, Ps, R)
    # Special case, no DIIS
    (diis.m == 0) && return (Pd, Ps)

    # First iteration
    if depth(diis) < 2
        push!(diis, xₙ, Rₙ)
        return xₙ
    end

    # Subsequant iterations
    𝐗 = diis.iterates
    𝐑 = diis.residuals
    T = eltype(xₙ)
    
    # Solve DIIS extrapolation system
    N_eq = depth(diis) + 1
    𝐒 = ones(T, N_eq, N_eq); 𝐒[1:end-1, 1:end-1] .= 𝐑'𝐑
    𝚪 = zeros(T, N_eq); 𝚪[end] = one(T)
    @show α = 𝐒\𝚪

    # Assemble new point
    x_diis = sum(α[1:end-1] .* 𝐗)
    x_diis = reshape(x_diis, size(xₙ))
    (test_MOs(x_diis, info.ζ.M.mo_numbers) > 1e-8) && (@warn "DIIS MOs may be too far from"*
                                                       " the Manifold. Try launching DIIS closer to a minimum")
    x_diis
end
