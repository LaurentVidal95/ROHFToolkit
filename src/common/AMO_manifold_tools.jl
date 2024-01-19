function retract_AMO(Ψ::Matrix{T}, Φ::Matrix{T}) where {T<:Real}
    # First compute the B matrix such that Ψ=ΦB
    B = Φ'Ψ
    ret = Φ*exp(B)
    @assert norm(ret'ret - I) < 1e-8
    ret
end
retract_AMO(ζ, Ψ::Matrix, Φ::Matrix) = retract_AMO(Ψ, Φ)

function retract_AMO(Ψ::TangentVector)
    @assert Ψ.base.isortho
    RΨ = retract_AMO(Ψ.vec, Ψ.base.Φ)
    State(Ψ.base, RΨ)
end

"""
Transport of η1 along η2 from ζ to Rζ(η2)
"""
function transport_AMO_same_dir(η::TangentVector{T}, α::T, Rη::State{T}) where {T<:Real}
    TangentVector(α*η.vec, Rη)
end
