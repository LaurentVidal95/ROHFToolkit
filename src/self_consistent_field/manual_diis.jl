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
function Base.push!(diis::DIIS, Pdₙ, Psₙ, Rₙ)
    if depth(diis) + 1 > diis.m
        norms = norm.(M.residuals)
        _, idx = findmax(norms)
        deleteat!(diis.iterates)
        deleteat!(diis.residuals)
    end
    push!(diis.iterates,  deepcopy(hcat(vec(Pdₙ), vec(Psₙ))))
    push!(diis.residuals, deepcopy(hcat(vec.(Rₙ)...)))
    @assert depth(diis) <= diis.m + 1
    @assert length(diis.residuals) == depth(diis)
end

function (diis::DIIS)(info)
    Pdₙ, Psₙ = info.DMs
    Rₙ = info.∇E

    # Special case, no DIIS
    (diis.m == 0) && return (Pdₙ, Psₙ)

    # First iteration
    if depth(diis) < 2
        push!(diis, Pdₙ, Psₙ, Rₙ)
        return Pdₙ, Psₙ
    end

    # Subsequant iterations
    𝐏 = diis.iterates
    𝐑 = diis.residuals
    T = eltype(Pdₙ)
    
    # Solve DIIS extrapolation system
    N_eq = depth(diis) + 1
    𝐒 = ones(T, N_eq, N_eq); 𝐒[1:end-1, 1:end-1] .= 𝐑'𝐑
    𝚪 = zeros(T, N_eq); 𝚪[end] = one(T)
    @show α = 𝐒\𝚪

    # Assemble new point
    Pds_diis = sum(α[1:end-1] .* 𝐏)
    (Nb, Nd, Ns) = info.ζ.Σ.mo_numbers
    Pd_diis = reshape(Pds_diis[1:Nb*Nb], (Nb, Nb))
    Ps_diis = reshape(Pds_diis[Nb*Nb+1:end],(Nb, Nb))
    Pd_diis, Ps_diis
end
