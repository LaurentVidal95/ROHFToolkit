"""
Standard fixed depth DIIS as described e.g. in [https://doi.org/10.1051/m2an/2021069]
(Its a review, see Pulay for the introduction of DIIS I think)
DIIS is applied in DM conventions to match the ROHF paper writing.
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
    push!(diis.iterates,  (Pdₙ, Psₙ))
    push!(diis.residuals, vec(Rₙ))
    if depth(diis) > diis.m + 1
        popfirst!(diis.iterates)
        popfirst!(diis.residuals)
    end
    @assert depth(diis) <= diis.m + 1
    @assert length(diis.residuals) == depth(diis)
    diis
end

function (diis::DIIS)(info)
    Pdₙ, Psₙ = info.DMs
    Rₙ = info.∇E

    # Special case, no DIIS
    (diis.m == 0) && return (Pdₙ, Psₙ)
    # First iteration
    if isempty(diis.iterates)
        push!(diis, Pdₙ, Psₙ, Rₙ)
        return Pdₙ, Psₙ
    end

    push!(diis, Pdₙ, Psₙ, Rₙ)
    𝐏 = diis.iterates
    𝐑 = diis.residuals
    𝐒 = hcat([𝐑[i+1] - 𝐑[i] for i in 1:length(𝐑)-1]...)

    # Solve DIIS least square PB
    A = 𝐒'𝐒
    B = 𝐒'𝐑[end]
    C = A\B

    𝐏d_diff = [𝐏[i+1][1] - 𝐏[i][1] for i in 1:length(𝐏)-1]
    𝐏s_diff = [𝐏[i+1][2] - 𝐏[i][2] for i in 1:length(𝐏)-1]
    Pd_diis = Pdₙ - C'𝐏d_diff
    Ps_diis = Psₙ - C'𝐏s_diff
    Pd_diis, Ps_diis
end
