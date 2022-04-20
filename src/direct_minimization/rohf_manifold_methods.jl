function retract(M::ROHFManifold, Ψ::ROHFTangentVector{T}) where {T<:Real}
    Nb, Nd, Ns = M.mo_numbers
    No = Nd+Ns
    (Ψd, Ψs), (Φd, Φs) = split_MOs(Ψ)

    # d <-> s rotations
    X = -Φd'Ψs
    W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';

    # occupied <-> virtual
    Ψd_tilde = Ψd .- Φd*Φd'*Ψd .- Φs*Φs'*Ψd;
    Ψs_tilde = Ψs .- Φd*Φd'*Ψs .- Φs*Φs'*Ψs;
    V1,D,V2 = svd(hcat(Ψd_tilde, Ψs_tilde))
    Σ = diagm(D)

    ROHFState(Ψ.foot, (Ψ.foot.Φ*V2*cos(Σ) + V1*sin(Σ))*V2' * exp(W))
end
function retract!(M::ROHFManifold{T}, Ψ::ROHFTangentVector{T}) where {T<:Real}
    Ψ.vec = retract(M, Ψ).Φ
    nothing
end

"""
For (Ψd,Ψs) in R^{Nb×Nd}×R^{Nb×Ns} and y = (Φd,Φs) in the MO manifold
the orthogonal projector on the horizontal tangent space at y is defined by

Π_y(Ψd|Ψs) = ( 1/2*Φs[Φs'Ψd - Ψs'Φd] + Φv(Φv'Ψd) | -1/2*Φd[Ψd'Φs - Φd'Ψs] + Φv(Φv'Ψs) )

If Φ is no the foot of Ψ, may serve as an alternative to transport 
Ψ in the tangent plane to Φ.
"""
function project_tangent(Φ::Matrix{T}, Ψ::Matrix{T}, mo_numbers) where {T<:Real}
    Ψd, Ψs = split_MOs(Ψ, mo_numbers)
    Φd, Φs = split_MOs(Φ, mo_numbers)
    X = 1/2 .* (Ψd'Φs + Φd'Ψs);
    hcat(-Φs*X' + (I - Φd*Φd')*Ψd, -Φd*X + (I - Φs*Φs')*Ψs)
end
project_tangent(M::ROHFManifold, Φ::Matrix, Ψ::Matrix) =
    project_tangent(Φ, Ψ, M.mo_numbers)
function project_tangent!(M::ROHFManifold, Ψ::ROHFTangentVector)
    Ψ.vec = project_tangent(M, Ψ.foot, Ψ.vec)
    nothing
end

"""
Transport a directin p along t*p. Used as default transport for SD and CG
in the linesearch routine on the ROHF manifold.
"""
function transport_vec_along_himself(Ψ::ROHFTangentVector{T}, t::T,
                                     ζ_next::ROHFState{T}) where {T<:Real}
    # Check that the targeted point is in orthonormal AO convention
    @assert (ζ_next.isortho)

    Nb, Nd, Ns = ζ_next.M.mo_numbers
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

    τ_p = (-Ψ.foot.Φ*V2*sin(t .*Σ) + V1*cos( t .*Σ))*Σ*V2' * exp(t .* W) + ζ_next.Φ*W
    Ξd, Ξs = split_MOs(τ_p, (Nb,Nd,Ns))
    ROHFTangentVector(hcat(Ξd - Φd_next*Φd_next'Ξd, Ξs - Φs_next*Φs_next'Ξs), ζ_next)
end
