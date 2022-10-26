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

function Fock_operators(Pdᵒ, Psᵒ, ζ::ROHFState{T}) where {T<:Real}
    _, eri, H = collect(ζ)[1:end-2]
    Sm12 = ζ.Σ.Sm12
    Pd = Symmetric(Sm12*Pdᵒ*Sm12); Ps = Symmetric(Sm12*Psᵒ*Sm12)
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    Fock_operators(Jd, Js, Kd, Ks, H, Sm12)
end

function gradient_DM_metric(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T}, ζ::ROHFState{T}) where {T<:Real}
    # ortho -> non-ortho densities
    Fdᵒ, Fsᵒ = Fock_operators(Pdᵒ, Psᵒ, ζ)
    project_tangent_DM(Pdᵒ, Psᵒ, 2*Fdᵒ, 2*Fsᵒ)
end

function energy_and_gradient_DM_metric(Pdᵒ, Psᵒ, Sm12, mo_numbers, eri, H, mol)
    Pd, Ps = Symmetric(Sm12*Pdᵒ*Sm12), Symmetric(Sm12*Psᵒ*Sm12)
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    # energy
    E = energy(Pd, Ps, Jd, Js, Kd, Ks, H, mol)
    # gradient
    Fdᵒ, Fsᵒ = Fock_operators(Jd, Js, Kd, Ks, H, Sm12)
    ∇E = project_tangent_DM(Pdᵒ, Psᵒ, 2*Fdᵒ, 2*Fsᵒ)
    E, ∇E
end
function energy_and_gradient_DM_metric(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T}, ζ::ROHFState{T}) where {T<:Real}
    energy_and_gradient_DM_metric(Pdᵒ, Psᵒ, ζ.Σ.Sm12, collect(ζ)[1:end-1]...)
end

"""
Mapping from a tangent space at M_DM to a tangent space at M_MO.
"""
function TDM_to_TMO(η::ROHFTangentVector{T}) where {T<:Real}
    (Nb, Nd, Ns) = η.base.Σ.mo_numbers
    Φd, Φs = split_MOs(η.base)    
    Pd, Ps = Φd*Φd', Φs*Φs'
    Qd, Qs = η.vec[:, 1:Nb], η.vec[:,Nb+1:end]
    hcat( (I-Pd)*Qd*Φd, (I-Ps)*Qs*Φs )
end

