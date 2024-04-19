import Base: +, -, *, adjoint, vec, size
import LinearAlgebra.norm

@doc raw"""
     TangentVector(vec::AbstractMatrix{T}, base::State{T})

Tangent vectors of the AMO matrix manifold are of the form  ``X = Cκ``
where ``C`` is the foot of the vector, and ``κ`` is a ``N_b×N_b`` antisymmetric
matrix with zero I, A and E diagonal blocs.
"""
struct TangentVector{T<:Real}
    kappa::AbstractMatrix{T}
    base::State{T}
end

"""
Overloading basic operations for TangentVectors.
"""
(+)(A::Matrix, X::TangentVector) = (+)(A, X.kappa)
(+)(X::TangentVector, A::Matrix) = (+)(X.kappa, A)
(*)(λ::Real, X::TangentVector) = (*)(λ, X.kappa)
(*)(A::Adjoint{Float64, Matrix{Float64}}, X::TangentVector) = (*)(A, X.kappa)
(-)(X::TangentVector) = -X.kappa
norm(X::TangentVector) = norm(X.kappa)
(adjoint)(X::TangentVector) = X.kappa'
(vec)(X::TangentVector) = vec(X.kappa)

orthonormalize_state!(Ψ::TangentVector) = orthonormalize_state!(Ψ.base)
