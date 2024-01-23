@doc raw"""
TODO
"""
function remove_diag_blocs!(M::Matrix, mo_numbers)
    Nb, Ni, Na = mo_numbers
    Ne = Nb - (Ni+Na)
    No = Ni+Na

    M[1:Ni, 1:Ni] .= zeros(Ni, Ni)
    M[Ni+1:No, Ni+1:No] .= zeros(Na, Na)
    M[No+1:Nb, No+1:Nb] .= zeros(Ne, Ne)
    nothing
end

@doc raw"""
TODO
"""
function project_tangent_AMO(Φ::Matrix, mo_numbers, M::Matrix)
    # Construct B matrix and project
    B_proj = asym(Φ'M)
    remove_diag_blocs!(B_proj, mo_numbers)
    # Add Φ
    Φ*B_proj
end
function project_tangent_AMO(ζ::State, M::Matrix)
    @assert ζ.isortho
    @assert ζ.virtuals
    project_tangent_AMO(ζ.Φ, ζ.Σ.mo_numbers, M)
end

@doc raw"""
TODO
"""
function retract_AMO(Φ::Matrix{T}, Ψ::Matrix{T}) where {T<:Real}
    B = Φ'Ψ
    Φ*exp(B)
end
function retract_AMO(Ψ::TangentVector)
    @assert Ψ.base.isortho
    RΨ = retract_AMO(Ψ.base.Φ, Ψ.vec)
    State(Ψ.base, RΨ)
end

"""
Transport of η1 along η2 from ζ to Rζ(η2)
"""
function transport_colinear_AMO(η::TangentVector{T}, α::T, Rη::State{T}) where {T<:Real}
    B = η.base.Φ'η.vec
    TangentVector(Rη.Φ*B, Rη)
end

@doc raw"""
TODO
"""
function transport_non_colinear_AMO(η1::TangentVector{T}, ζ::State{T},
                                    η2::TangentVector{T}, α::T, Rη2::State{T}) where {T<:Real}
    # Assert that the two vectors live in the same initial tangent space
    @assert η1.base.Φ == η2.base.Φ

    # Extract antisymmetric matrices
    X = η1.base.Φ'*η1.vec
    B = η2.base.Φ'*η2.vec
    mo_numbers = ζ.Σ.mo_numbers

    # Compute the transport exponential post factor
    function exp_φ(X; tol=1e-8)
        # k = 0
        out = X
        # k = 1
        k = 1
        φ = compute_φ_transport(X, α .* B, mo_numbers)
        prefac = ((-1)^k)/factorial(k)
        next_term = prefac .* φ
        out = out .+ next_term
        # k > 1
        while norm(next_term) > tol
            k += 1
            prefac = ((-1)^k)/factorial(k)
            φ = compute_φ_transport(φ, B, mo_numbers)
            next_term = prefac .* φ
            out = out .+ next_term
        end
        out
    end

    # transport and return in a TangentVector struct
    τη1_vec = Rη2.Φ*exp_φ(X)
    TangentVector(τη1_vec, Rη2)
end
function compute_φ_transport(X::Matrix, B::Matrix, mo_numbers)
    commutator = 0.5 .* (B*X - X*B)
    remove_diag_blocs!(commutator, mo_numbers)
    commutator
end
