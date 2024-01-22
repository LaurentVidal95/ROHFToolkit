function retraction(ζ::State, Ψ::Matrix)
    # TODO make it work without virtuals    
    Nb, Ni, Na = ζ.Σ.mo_numbers
    @assert ζ.virtuals
    @assert ζ.isortho

    # Retract
    B = ζ.Φ'Ψ
    RΨ = ζ.Φ*exp(B)

    # Select all MOs or just occupied MOs
    !(ζ.virtuals) && (RΨ .= RΨ*Matrix(I, Nb, Ni+Na))
    State(ζ, RΨ)
end
retraction(Ψ::TangentVector) = retraction(Ψ.base, Ψ.vec)

function projection_tangent(ζ::State, M::Matrix)
    @assert ζ.isortho
    @assert ζ.virtuals
    Nb, Ni, Na = ζ.Σ.mo_numbers
    Ne = Nb - (Ni+Na)
    No = Ni+Na

    # Construct B matrix
    B_proj = asym(ζ.Φ'M)
    B_proj[1:Ni, 1:Ni] .= zeros(Ni, Ni)
    B_proj[Ni+1:No, Ni+1:No] .= zeros(Na, Na)
    B_proj[No+1:Nb, No+1:Nb] .= zeros(Ne, Ne)

    ζ.Φ*B_proj
end
function transport_collinear_vectors(η::TangentVector, t::T, ζ::State) where {T<:Real}
    @assert η.base.isortho
    @assert ζ.isortho
    Nb, Ni, Na = η.base.Σ.mo_numbers
    τη_vec = t*η
    #    !(ζ_next.virtuals) && (τη_vec .= τη_vec*Matrix(I, Nb, Ni+Na))
    ζ_next = retraction(ζ, t*η)
    τη = TangentVector(τη_vec, ζ_next)
end
