"""
All routines on the DM manifold. Used only for DIIS.
"""
#
# In this file the exponent ᵒ denote objects in orthonormal AOs convention
#
sym(A::AbstractMatrix) = Symmetric(0.5 .* (A + transpose(A)))
function project_tangent_DM(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T},
                            A::Matrix{T}, B::Matrix{T}) where {T<:Real}
    Pvᵒ = Symmetric(I - Pdᵒ - Psᵒ)
    Πdᵒ = sym(Pdᵒ*(A-B)*Psᵒ) + 2*sym(Pdᵒ*A*Pvᵒ)
    Πsᵒ = sym(Pdᵒ*(B-A)*Psᵒ) + 2*sym(Psᵒ*B*Pvᵒ)
    Πdᵒ, Πsᵒ
end

function gradient_DM_metric(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T}, ζ::ROHFState{T}) where {T<:Real}
    @assert(ζ.isortho)
    # ortho -> non-ortho densities
    Fdᵒ, Fsᵒ = Fock_operators(Pdᵒ, Psᵒ, ζ)
    hcat(project_tangent_DM(Pdᵒ, Psᵒ, 2*Fdᵒ, 2*Fsᵒ)...)
end

function energy_and_gradient_DM_metric(Pdᵒ, Psᵒ, Sm12, mo_numbers, eri, H, mol)
    Pd, Ps = Symmetric(Sm12*Pdᵒ*Sm12), Symmetric(Sm12*Psᵒ*Sm12)
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    # energy
    E = energy(Pd, Ps, Jd, Js, Kd, Ks, H, mol)
    # gradient
    Fdᵒ, Fsᵒ = Fock_operators(Jd, Js, Kd, Ks, H, Sm12)
    ∇E = hcat(project_tangent_DM(Pdᵒ, Psᵒ, 2*Fdᵒ, 2*Fsᵒ)...)
    E, ∇E
end
function energy_and_gradient_DM_metric(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T}, ζ::ROHFState{T}) where {T<:Real}
    energy_and_gradient_DM_metric(Pdᵒ, Psᵒ, ζ.Σ.Sm12, collect(ζ)[1:end-1]...)
end

# """
# Mapping from a tangent space at M_DM to a tangent space at M_MO.
# """
# function TDM_to_TMO(η::ROHFTangentVector{T}) where {T<:Real}
#     (Nb, Nd, Ns) = η.base.Σ.mo_numbers
#     Φd, Φs = split_MOs(η.base)    
#     Pd, Ps = Φd*Φd', Φs*Φs'
#     Qd, Qs = η.vec[:, 1:Nb], η.vec[:,Nb+1:end]
#     ROHFTangentVector(hcat( (I-Pd)*Qd*Φd, (I-Ps)*Qs*Φs ), η.base)
# end

# function TMO_to_TDM(η::ROHFTangentVector{T}) where {T<:Real}
#     (Ψd, Ψs), (Φd, Φs) = split_MOs(η)
#     Pd, Ps = project_tangent_DM(Φd*Φd', Φs*Φs', Φd*Ψd'+Ψd*Φd', Φs*Ψs'+Ψs*Φs')
#     ROHFTangentVector(hcat(Pd, Ps), η.base)
# end

# function retract_DM(ζ::ROHFState{T}, η_DM::ROHFTangentVector{T}) where {T<:Real}
#     @assert(η_DM.base.Φ == ζ.Φ) # check that ζ is the base of η
#     η_MO = TDM_to_TMO(η_DM)
#     Rη_MO, τη_MO = retract(ζ, η_MO, 1.)
#     # Switch to DM conventions
#     ROHFState(ζ, hcat(densities(Rη_MO)...)), TMO_to_TDM(τη_MO)
# end

# function transport_DM!(η1::ROHFTangentVector{T}, ζ::ROHFState{T},
#                        η2::ROHFTangentVector{T}, α::T, Rη2::ROHFState{T}) where {T<:Real}
#     (Nb, Nd, Ns) = ζ.Σ.mo_numbers
#     Pd1, Ps1 = η1.vec[:,1:Nb], η1.vec[:,Nb+1:end]
#     Pd2, Ps2 = η2.vec[:,1:Nb], η2.vec[:,Nb+1:end]
#     τη1_vec = hcat(project_tangent_DM(Pd1, Ps1, Pd2, Ps2)...)
#     ROHFTangentVector(τη1_vec, Rη2)
# end

# function precondition(ζ::ROHFState{T}, η) where {T<:Real}
    
# end

