@doc raw"""
TODO
"""
function project_tangent_AMO(Φ::Matrix, mo_numbers, M::Matrix)
    Nb, Ni, Na = mo_numbers
    Ne = Nb - (Ni+Na)
    No = Ni+Na

    # Construct B matrix
    B_proj = asym(Φ'M)
    B_proj[1:Ni, 1:Ni] .= zeros(Ni, Ni)
    B_proj[Ni+1:No, Ni+1:No] .= zeros(Na, Na)
    B_proj[No+1:Nb, No+1:Nb] .= zeros(Ne, Ne)

    Φ*B_proj
end
function project_tangent_AMO(ζ::State, M::Matrix)
    @assert ζ.isortho
    @assert ζ.virtuals
    project_tangent_AMO(ζ.Φ, ζ.Σ.mo_numbers, M)
end

@doc raw"""
TODO
TODO 2) Check numerical stability
"""
function retract_AMO(Φ::Matrix{T}, Ψ::Matrix{T}) where {T<:Real}
    B = Φ'Ψ
    Φ*exp(B)
end
#retract_AMO(ζ, Φ::Matrix, Ψ::Matrix) = retract_AMO(Ψ, Φ)

function retract_AMO(Ψ::TangentVector)
    @assert Ψ.base.isortho
    RΨ = retract_AMO(Ψ.base.Φ, Ψ.vec)
    State(Ψ.base, RΨ)
end

# """
# Transport of η1 along η2 from ζ to Rζ(η2)
# """
# function transport_AMO_same_dir(η::TangentVector{T}, α::T, Rη::State{T}) where {T<:Real}
#     TangentVector(α*η.vec, Rη)
# end
