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
    M
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
function exp_retract_AMO(Φ::Matrix{T}, Ψ::Matrix{T}) where {T<:Real}
    B = Φ'Ψ
    Φ*exp(B)
end
function QR_retract_AMO(Φ::Matrix{T}, Ψ::Matrix{T}) where {T<:Real}
   Matrix(qr(Φ + Ψ).Q)
end

function retract_AMO(Φ::Matrix{T}, Ψ::Matrix{T}; type=:exp) where {T<:Real}
    (type==:exp) && (return exp_retract_AMO(Φ, Ψ))
    (type==:QR) && (return QR_retract_AMO(Φ, Ψ))
    error("Given type of retraction not handled")
end

"""
Transport of η1 along η2 from ζ to Rζ(η2)
"""
function parallel_transport_collinear_AMO(η::TangentVector{T}, α::T, Rη::State{T}) where {T<:Real}
    B = η.base.Φ'η.vec
    TangentVector(Rη.Φ*B, Rη)
end

@doc raw"""
TODO
"""
function parallel_transport_non_collinear_AMO(η1::TangentVector{T}, ζ::State{T},
                                              η2::TangentVector{T}, α::T, Rη2::State{T}) where {T<:Real}
    # Assert that the two vectors live in the same initial tangent space
    @assert η1.base.Φ == η2.base.Φ

    # Extract antisymmetric matrices
    X = η1.base.Φ'*η1.vec
    B = η2.base.Φ'*η2.vec
    mo_numbers = ζ.Σ.mo_numbers

    # Compute the transport exponential post factor
    function exp_φ(X; tol=1e-8, kmax=20)
        ad_Bm(X) = remove_diag_blocs!(B*X - X*B, mo_numbers)
        # k = 0
        output = X
        # k = 1
        k=1
        current_term = - (α/2)*ad_Bm(X)
        output = output + current_term
        # k > 1
        while (norm(current_term) > tol) && (k < kmax)
            k += 1
            current_term = (-α/(2*k))*ad_Bm(current_term)
            output = output + current_term
        end
        if norm(current_term) > tol
            @warn "Transport trunctated before tolerance is reached"
            @show norm(current_term)
        end
        output
    end
    # transport and return in a TangentVector struct
    τη1_vec = Rη2.Φ*exp_φ(X)
    TangentVector(τη1_vec, Rη2)
end
function parallel_transport_AMO(η1::TangentVector{T}, ζ::State{T},
                                η2::TangentVector{T}, α::T, Rη2::State{T};
                                collinear=false) where {T<:Real}
    collinear && (return parallel_transport_collinear_AMO(η1, α, Rη2))
    return  parallel_transport_non_collinear_AMO(η1, ζ, η2, α, Rη2)
end

function QR_transport_non_collinear_AMO(Y::TangentVector{T}, x::State{T},
                                        X::TangentVector{T}, α::T, RX::State{T}) where {T<:Real}

    @assert X.base.Φ == Y.base.Φ
    # Renaming for clarity
    Φ = x.Φ
    # RΦ = RX.Φ
    RX_manual = State(x, Matrix(qr(x.Φ + α*X.vec).Q))
    RΦ = RX_manual.Φ
    # Preliminary computations
    M = inv(RΦ'*(Φ+X.vec))
    function ρ_skew(M)
        n, m = size(M)
        @assert n==m
        A = zero(M)
        for j in 1:n
            for i in j+1:n
                A[i,j] = M[i,j]
            end
        end
        A - A'
    end
    # return  (I-RΦ*RΦ')*Y.vec*M + RΦ*ρ_skew(RΦ'*Y.vec*M)
    τY_B =  RΦ'*( (I-RΦ*RΦ')*Y.vec*M + RΦ*ρ_skew(RΦ'*Y.vec*M))
    TangentVector(RΦ*remove_diag_blocs!(τY_B, x.Σ.mo_numbers), RX)
end
function QR_transport_AMO(args...;  collinear=false)
    # if colinear
    #     return QR_transport_colinear_AMO(TODO...)
    # end
    # For now no collinear transport for QR
    return  QR_transport_non_collinear_AMO(args...)
end

function transport_AMO(η1::TangentVector{T}, ζ::State{T},
                       η2::TangentVector{T}, α::T, Rη2::State{T};
                       type=:exp, collinear=false) where {T<:Real}
    (type==:exp) && (return parallel_transport_AMO(η1, ζ, η2, α, Rη2; collinear))
    (type==:QR) && (return QR_transport_AMO(η1, ζ, η2, α, Rη2; collinear))
    (type==:proj) && (return TangentVector(project_tangent_AMO(Rη2, η1.vec), Rη2))
    error("Given type of tranport not handled")
end
