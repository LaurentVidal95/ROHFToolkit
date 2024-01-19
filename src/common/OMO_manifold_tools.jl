@doc raw"""
    retract_OMO(mo_numbers::Tuple(Int64,Int64,Int64), Ψ::Matrix{T}, Φ::Matrix{T}) where {T<:Real}

Retraction on the ROHF flag manifold in MO coordinates that doesn't use virtual orbitals.
   - mo_numbers: tuple (Nb, Nd, Ns), respectively the total number, the number
        of doubly-occupied and the number singly occupied MOs.
   - ``Φ=[Φ_d|Φ_s]``: Matrix of MO coefficients, representing a point on the MO manifold
   - ``Ψ=[Ψ_d|Ψ_s]``: Matrix representing a point on the tangent space at Φ to the MO manifold

The exact formula of this retraction is given in equations (17) of the documentation.
"""
function retract_OMO(mo_numbers::Tuple{Int64, Int64, Int64},
                     Ψ::Matrix{T}, Φ::Matrix{T}) where {T<:Real}
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

    (Φ*V2*cos(Σ) + V1*sin(Σ))*V2' * exp(W)
end

"""
Same retraction routines in compressed format
"""
retract_OMO(ζ::State, Ψ, Φ) = retract_OMO(ζ.Σ.mo_numbers, Ψ, Φ)
function retract_OMO(Ψ::TangentVector)
    RΨ = retract_OMO(Ψ.base, Ψ.vec, Ψ.base.Φ)
    State(Ψ.base, RΨ)
end

@doc raw"""
    enforce_retract(Ψ::Matrix{T}) where {T<:Real}

Very unstable MO retraction. For a given matrix Ψ, compute the density
 ``P=Ψ Ψ^T``, diagonalize and set the eigenvalues close to 1 to exactly 1, 
and those close to 0 to exactly 0.  Only works if Ψ is close to the MO manifold.
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

@doc raw""" 
    project_tangent(mo_numbers::Tuple{Int64, Int64, Int64}, Φ::Matrix{T},
                         Ψ::Matrix{T}) where {T<:Real}

For a point ``Φ=[Φ_d|Φ_s]`` on the MO manifold and point
``[Ψd|Ψs]`` in ``\mathbb{R}^{Nb×Nd}×\mathbb{R}^{Nb×Ns}`` the orthogonal
projector on the horizontal tangent space at ``Φ`` is defined by
```math
    Π_Φ(Ψ) = [1/2*Φ_s(Φ_s^{T} Ψ_d - Ψ_s^{T} Φ_d) + Φ_v(Φ_v^{T} Ψ_d) |
         -\frac{1}{2} Φ_d(Ψ_d^{T} Φ_s - Φ_d^{T} Ψ_s] + Φ_v(Φ_v^{T} Ψ_s) ]
```
If Φ is not the base of Ψ, serve as the default (because the cheapest) way to transport
Ψ in the tangent plane to Φ.
"""
function project_tangent(mo_numbers::Tuple{Int64, Int64, Int64}, Φ::Matrix{T},
                         Ψ::Matrix{T}) where {T<:Real}
    Ψd, Ψs = split_MOs(Ψ, mo_numbers)
    Φd, Φs = split_MOs(Φ, mo_numbers)
    X = 1/2 .* (Ψd'Φs + Φd'Ψs);
    hcat(-Φs*X' + (I - Φd*Φd')*Ψd, -Φd*X + (I - Φs*Φs')*Ψs)
end
project_tangent(ζ::State, Φ::Matrix, Ψ::Matrix) =
    project_tangent(ζ.Σ.mo_numbers, Φ, Ψ)


@doc raw"""
    transport_vec_along_himself_OMO(Ψ::TangentVector{T}, t::T,
                                     ζ_next::State{T}) where {T<:Real}

Transport a direction Ψ along t*Ψ. Used as an alternative transport to
orthogonal projection for SD and CG in the linesearch routine on
the ROHF manifold.

The formula for this transport is given in the equation (24) of
the documentation.
"""
function transport_OMO_same_dir(Ψ::TangentVector{T}, t::T,
                                ζ_next::State{T}) where {T<:Real}
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
    TangentVector(hcat(Ξd - Φd_next*Φd_next'Ξd, Ξs - Φs_next*Φs_next'Ξs), ζ_next)
end

## Retraction using geodesic (i.e. virtual orbitals) for numerical tests.
# function retract(mo_numbers::Tuple{Int64, Int64, Int64}, Ψ::Matrix{T}, Φ::Matrix{T}) where {T<:Real}
#     Nb, Nd, Ns = mo_numbers
#     No = Nd+Ns

#     Ψd, Ψs = split_MOs(Ψ, (Nb,Nd,Ns))
#     Φd, Φs = split_MOs(Φ, (Nb,Nd,Ns))
#     Φv = generate_virtual_MOs(Φd, Φs, (Nb,Nd,Ns))
#     Φ_tot = hcat(Φd, Φs, Φv)

#     X = -Φd'Ψs
#     Y = Φv'Ψd
#     Z = Φv'Ψs
#     W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';
    
#     B = zeros(Nb, Nb)
#     B[1:Nd, Nd+1:Nd+Ns] = -X;
#     B[1:Nd, Nd+Ns+1:Nb] = -Y';
#     B[Nd+1:Nd+Ns, Nd+Ns+1:Nb] = -Z';
#     B = B - B'
    
#     (Φ_tot * exp(B)) * Matrix(I, Nb, No)
# end
# function retract(mo_numbers::Tuple{Int64, Int64, Int64}, Ψ::Matrix{T}, Φ::Matrix{T}) where {T<:Real}
#     V1,D,V2 = svd(Ψ)
#     Σ = diagm(D)
#     (Φ*V2*cos(Σ) + V1*sin(Σ))*V2'
# end
