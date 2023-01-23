#
# Retraction and vector transports on the ROHF manifold.
#
function retract(mo_numbers::Tuple{Int64, Int64, Int64}, Ψ::Matrix{T}, Φ::Matrix{T}) where {T<:Real}
    Nb, Nd, Ns = mo_numbers
    No = Nd+Ns
    Ψd, Ψs = split_MOs(Ψ, (Nb,Nd,Ns))
    Φd, Φs = split_MOs(Φ, (Nb,Nd,Ns))

    # d <-> s rotations
    X = -Φd'Ψs
    W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';

    # occupied <-> virtual
    Ψd_tilde = Ψd .- Φd*Φd'*Ψd .- Φs*Φs'*Ψd;
    Ψs_tilde = Ψs .- Φd*Φd'*Ψs .- Φs*Φs'*Ψs;
    V1,D,V2 = svd(hcat(Ψd_tilde, Ψs_tilde))
    Σ = diagm(D)

    ret = (Φ*V2*cos(Σ) + V1*sin(Σ))*V2' * exp(W)
    @show test_MOs(ret, mo_numbers)
    ret
end
# function retract(mo_numbers::Tuple{Int64, Int64, Int64}, Ψ::Matrix{T}, Φ::Matrix{T}) where {T<:Real}
#     V1,D,V2 = svd(Ψ)
#     Σ = diagm(D)

#     Φ_ret = (Φ*V2*cos(Σ) + V1*sin(Σ))*V2'
#     @show test_MOs(Φ_ret, mo_numbers)
#     Φ_ret
# end

retract(ζ::ROHFState, Ψ, Φ) = retract(ζ.Σ.mo_numbers, Ψ, Φ)

function retract(Ψ::ROHFTangentVector)
    RΨ = retract(Ψ.base, Ψ.vec, Ψ.base.Φ)
    ROHFState(Ψ.base, RΨ)
end

"""
Very unstable retraction
"""
function enforce_retract(Ψ::AbstractMatrix{T}) where {T<:Real}
    P = Ψ*Ψ'
    No = round(Int,tr(P))
    Io = diagm(ones(No))
    U = eigen(-Symmetric(P)).vectors
    P_ret = Symmetric(U[:,1:No]*Io*U[:,1:No]')
    @warn "forced restraction"
    eigen(Symmetric(P_ret)).vectors[:,1:No]
end

"""
For (Ψd,Ψs) in R^{Nb×Nd}×R^{Nb×Ns} and y = (Φd,Φs) in the MO manifold
the orthogonal projector on the horizontal tangent space at y is defined by

Π_y(Ψd|Ψs) = ( 1/2*Φs[Φs'Ψd - Ψs'Φd] + Φv(Φv'Ψd) | -1/2*Φd[Ψd'Φs - Φd'Ψs] + Φv(Φv'Ψs) )

If Φ is not the base of Ψ, may serve as an alternative to transport
Ψ in the tangent plane to Φ.
"""
function project_tangent(mo_numbers::Tuple{Int64, Int64, Int64}, Φ::Matrix{T},
                         Ψ::Matrix{T}) where {T<:Real}
    Ψd, Ψs = split_MOs(Ψ, mo_numbers)
    Φd, Φs = split_MOs(Φ, mo_numbers)
    X = 1/2 .* (Ψd'Φs + Φd'Ψs);
    hcat(-Φs*X' + (I - Φd*Φd')*Ψd, -Φd*X + (I - Φs*Φs')*Ψs)
end
project_tangent(ζ::ROHFState, Φ::Matrix, Ψ::Matrix) =
    project_tangent(ζ.Σ.mo_numbers, Φ, Ψ)

"""
Transport a direction p along t*p. Used as default transport for SD and CG
in the linesearch routine on the ROHF manifold.
"""
function transport_vec_along_himself(Ψ::ROHFTangentVector{T}, t::T,
                                     ζ_next::ROHFState{T}) where {T<:Real}
    # Check that the targeted point is in orthonormal AO convention
    @assert (ζ_next.isortho)

    Nb, Nd, Ns = ζ_next.Σ.mo_numbers
    No = Nd+Ns
    (Ψd, Ψs), (Φd, Φs) = split_MOs(Ψ)
    Φd_next, Φs_next = split_MOs(ζ_next)

    # d <-> s rotations
    X = -Φd'*Ψs;
    W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';

    # occupied <-> virtual
    Ψd_tilde = Ψd .- Φd*Φd'*Ψd .- Φs*Φs'*Ψd;
    Ψs_tilde = Ψs .- Φd*Φd'*Ψs .- Φs*Φs'*Ψs;
    V1,D,V2 = svd(hcat(Ψd_tilde, Ψs_tilde))
    Σ = diagm(D)

    τ_p = (-Ψ.base.Φ*V2*sin(t .*Σ) + V1*cos( t .*Σ))*Σ*V2' * exp(t .* W) + ζ_next.Φ*W
    Ξd, Ξs = split_MOs(τ_p, (Nb,Nd,Ns))
    ROHFTangentVector(hcat(Ξd - Φd_next*Φd_next'Ξd, Ξs - Φs_next*Φs_next'Ξs), ζ_next)
end
