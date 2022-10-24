"""
Standard fixed depth DIIS as described e.g. in [https://doi.org/10.1051/m2an/2021069]
(Its a review, see Pulay for the introduction of DIIS I think)
"""
mutable struct DIIS
    m::Int                            # maximal history size
    d_densities::Vector{Any}             # [Pₙ, Pₙ₋₁, ...]
    s_densities::Vector{Any}
    residuals::Vector{Any}            # [rₙ, rₙ₋₁, ...]
end
DIIS(;m=10) = DIIS(m, [], [], [])

# Actualize diis lists and remove old iterates if needed
function Base.push!(diis::DIIS, Pdₙ, Psₙ, Rₙ)
    push!(diis.d_densities,  vec(Pdₙ))
    push!(diis.s_densities, vec(Psₙ))
    push!(diis.residuals, vec(Rₙ))
    if length(diis.d_densities) > diis.m + 1
        popfirst!(diis.d_densities)
        popfirst!(diis.s_densities)
        popfirst!(diis.residuals)
    end
    @assert length(diis.d_densities) <= diis.m + 1
    @assert length(diis.d_densities) == length(diis.residuals)
    diis
end

function (diis::DIIS)(xₙ, Rₙ; info)
    Nb, Nd, Ns = info.ζ.M.mo_numbers

    # Special case, no DIIS
    (diis.m == 0) && return xₙ

    # First iteration
    if isempty(diis.d_densities)
        Pdₙ, Psₙ = densities(xₙ, (Nb, Nd, Ns))
        push!(diis, Pdₙ, Psₙ, Rₙ)
        return xₙ
    end

    # Subsequant iterations
    Pdₙ, Psₙ = densities(xₙ, (Nb, Nd, Ns)) 
    push!(diis, Pdₙ, Psₙ, Rₙ)
    𝐏d = diis.d_densities
    𝐏s = diis.s_densities
    𝐑 = diis.residuals

    # Solve DIIS least square pb on densities
    # Rq: We consider densities because the DIIS point on MOs
    # is too far from the manifold.
    𝐒 = hcat([𝐑[i+1] - 𝐑[i] for i in 1:length(𝐑)-1]...)
    A = 𝐒'𝐒
    B = 𝐒'vec(Rₙ)
    α = A\B

    # Assemble next point
    𝐏d_diff = hcat([𝐏d[i+1] - 𝐏d[i] for i in 1:length(𝐏d)-1]...)
    𝐏s_diff = hcat([𝐏s[i+1] - 𝐏s[i] for i in 1:length(𝐏s)-1]...)
    Pdₙ₊₁ = vec(Pdₙ) - sum(eachcol(α' .* 𝐏d_diff))
    Psₙ₊₁ = vec(Psₙ) - sum(eachcol(α' .* 𝐏s_diff))
    Pdₙ₊₁ = reshape(Pdₙ₊₁, Nb, Nb); Psₙ₊₁ = reshape(Psₙ₊₁, Nb, Nb)
    xₙ₊₁ = hcat(eigen(-Symmetric(Pdₙ₊₁)).vectors[:,1:Nd],
                eigen(-Symmetric(Psₙ₊₁)).vectors[:,1:Ns])
    if !isempty(info)
        (test_MOs(xₙ₊₁, (Nb,Nd,Ns)) > 1e-8) && (@warn "DIIS MOs may be too far from"*
                                 " the Manifold. Try launching DIIS closer to a minimum")
    end
    xₙ₊₁
end
