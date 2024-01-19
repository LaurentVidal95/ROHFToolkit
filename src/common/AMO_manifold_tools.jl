function retract_AMO(Ψ::Matrix{T}, Φ::Matrix{T}) where {T<:Real}
    # First compute the B matrix such that Ψ=ΦB
    B = Φ'Ψ
    Φ*exp(B)
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
function transport_AMO(η1::TangentVector{T}, ζ::State{T},
                       η2::TangentVector{T}, α::T, Rη2::State{T}) where {T<:Real}
    @assert ζ.isortho
    Nb, Ni, Na = ζ.Σ.mo_numbers
    Φ = ζ.Φ
    B = α*Φ'η2.vec # direction of transport
    X = Φ'η1.vec # transported vector

    Bia = B[1:Ni, Ni+1:Ni+Na]
    Bie = B[1:Ni, Ni+Na+1:Nb-(Ni+Na)]
    Bae = B[Ni+1:Ni+Na, Ni+Na+1:Nb-(Ni+Na)]
    Xia = X[1:Ni, Ni+1:Ni+Na]
    Xie = X[1:Ni, Ni+Na+1:Nb-(Ni+Na)]
    Xae = X[Ni+1:Ni+Na, Ni+Na+1:Nb-(Ni+Na)]
    
    # Compute the φ matrix
    U = -Bia*Xae'+ Xia*Bae'
    φ = zeros(Nb, Nb)
    φ[1:Ni, Ni+1:Ni+Na] .= U
    φ[Ni+1:Ni+Na, 1:Ni] .= -U'

    τη1 = Φ * exp(B)*exp(-φ)*X
    TangentVector(τη1, Rη2)
end
