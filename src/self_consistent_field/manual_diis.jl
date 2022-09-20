struct AndersonAcceleration
    m::Int                  # maximal history size
    iterates::Vector{Any}   # xₙ
    residuals::Vector{Any}  # Pf(xₙ)
    maxcond::Real           # Maximal condition number for Anderson matrix
end
AndersonAcceleration(;m=10, maxcond=1e6) = AndersonAcceleration(m, [], [], maxcond)

function Base.push!(anderson::AndersonAcceleration, xₙ, αₙ, Pfxₙ)
    push!(anderson.iterates,  vec(xₙ))
    push!(anderson.residuals, vec(Pfxₙ))
    if length(anderson.iterates) > anderson.m
        popfirst!(anderson.iterates)
        popfirst!(anderson.residuals)
    end
    @assert length(anderson.iterates) <= anderson.m
    @assert length(anderson.iterates) == length(anderson.residuals)
    anderson
end

# Gets the current xₙ, Pf(xₙ) and damping αₙ
# function (anderson::AndersonAcceleration)(xₙ, αₙ, Pfxₙ)
#     xs   = anderson.iterates
#     Pfxs = anderson.residuals

#     # Special cases with fast exit
#     anderson.m == 0 && return xₙ
#     if isempty(xs)
#         push!(anderson, xₙ, αₙ, Pfxₙ)
#         return xₙ
#     end

#     M = hcat(Pfxs...) .- vec(Pfxₙ)  # Mᵢⱼ = (Pfxⱼ)ᵢ - (Pfxₙ)ᵢ
#     # We need to solve 0 = M' Pfxₙ + M'M βs <=> βs = - (M'M)⁻¹ M' Pfxₙ

#     # Ensure the condition number of M stays below maxcond, else prune the history
#     Mfac = qr(M)
#     while size(M, 2) > 1 && cond(Mfac.R) > anderson.maxcond
#         M = M[:, 2:end]  # Drop oldest entry in history
#         popfirst!(anderson.iterates)
#         popfirst!(anderson.residuals)
#         Mfac = qr(M)
#     end

#     xₙ₊₁ = vec(xₙ) .+ αₙ .* vec(Pfxₙ)
#     βs   = -(Mfac \ vec(Pfxₙ))
#     for (iβ, β) in enumerate(βs)
#         xₙ₊₁ .+= β .* (xs[iβ] .- vec(xₙ) .+ αₙ .* (Pfxs[iβ] .- vec(Pfxₙ)))
#     end

#     push!(anderson, xₙ, αₙ, Pfxₙ)
#     reshape(xₙ₊₁, size(xₙ))
# end
