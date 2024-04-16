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
# function project_tangent_AMO(Φ::Matrix, mo_numbers, M::Matrix)
#     # Construct κ matrix and project
#     κ_proj = asym(Φ'M)
#     remove_diag_blocs!(κ_proj, mo_numbers)
# end
# function project_tangent_AMO(ζ::State, M::Matrix)
#     @assert ζ.isortho
#     @assert ζ.virtuals
#     κ = project_tangent_AMO(ζ.Φ, ζ.Σ.mo_numbers, M)
#     TangentVector(κ, ζ)
# end

function ensure_tangent_AMO(X::TangentVector)
    x = X.base
    κ_proj = copy((1/2)*(X.kappa - X.kappa'))
    remove_diag_blocs!(κ_proj, x.Σ.mo_numbers)
    TangentVector(κ_proj, x)
end

@doc raw"""
TODO
"""
exp_retract_AMO(Φ::Matrix{T}, κ::Matrix{T}, α=1.) where {T<:Real} =
    Φ*exp(α*κ)
QR_retract_AMO(Φ::Matrix{T}, κ::Matrix{T}, α=1.) where {T<:Real} =
   Φ*Matrix(qr(I + α*κ).Q)

function retract_AMO(ζ::State{T}, η::TangentVector{T}, α=1; type=:exp) where {T<:Real}
    Rη = zero(ζ.Φ)
    (type==:exp) && (Rη = exp_retract_AMO(ζ.Φ, η.kappa, α))
    (type==:QR)  && (Rη = QR_retract_AMO(ζ.Φ, η.kappa, α))
    State(ζ, Rη)
end

"""
Transport of η1 along η2 from ζ to Rζ(η2)
"""
parallel_transport_collinear_AMO(η::TangentVector{T}, α::T, Rη::State{T}) where {T<:Real} =
    TangentVector(η.kappa, Rη)

@doc raw"""
TODO
"""
function parallel_transport_non_collinear_AMO(η1::TangentVector{T}, ζ::State{T},
                                              η2::TangentVector{T}, α::T, Rη2::State{T}
                                              ) where {T<:Real}
    # Assert that the two vectors live in the same initial tangent space
    @assert η1.base.Φ == η2.base.Φ

    # Extract antisymmetric matrices
    κ1 = η1.kappa
    κ2 = η2.kappa
    mo_numbers = ζ.Σ.mo_numbers

    # Compute the transport exponential post factor
    function exp_φ(κ1; tol=eps(T), kmax=200)
        ad_κ2m(κ1) = remove_diag_blocs!(κ2*κ1 - κ1*κ2, mo_numbers)
        # k = 0
        output = κ1
        # k = 1
        k=1
        current_term = - (α/2)*ad_κ2m(κ1)
        output = output + current_term
        # k > 1
        while (norm(current_term) > tol) && (k < kmax)
            k += 1
            current_term = (-α/(2*k))*ad_κ2m(current_term)
            output = output + current_term
        end
        if norm(current_term) > tol
            @warn "Transport trunctated before tolerance is reached.\n"*
                "Norm of the last term $(norm(current_term)).\n"*
                "You might want to reduce the maximum step size."
        end
        output
    end
    # transport and return in a TangentVector structure
    TangentVector(exp_φ(κ1), Rη2)
end
function parallel_transport_AMO(η1::TangentVector{T}, ζ::State{T},
                                η2::TangentVector{T}, α::T, Rη2::State{T};
                                collinear=false) where {T<:Real}
    collinear && (return parallel_transport_collinear_AMO(η1, α, Rη2))
    return  parallel_transport_non_collinear_AMO(η1, ζ, η2, α, Rη2)
end

@doc raw"""
Simply projects η1 on the tangent space to Rη2
"""
function projection_transport_AMO(η1::TangentVector{T}, ζ::State{T}, η2::TangentVector{T},
                                  α::T, Rη2::State{T}) where {T<:Real}
    project_tangent_AMO(Rη2, η1.kappa)
end

function QR_transport_non_collinear_AMO(Y::TangentVector{T}, x::State{T},
                                        X::TangentVector{T}, α::T, RX::State{T}) where {T<:Real}
    
    error("Adapt to new TangentVector convention")
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
    (type==:proj) && (return projection_transport_AMO(η1, ζ, η2, α, Rη2))
    error("Given type of tranport not handled")
end
