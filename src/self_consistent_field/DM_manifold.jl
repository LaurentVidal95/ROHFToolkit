# All routines on the DM manifold. Used only for the residual in SCF routines.
# In this file the exponent ᵒ denote objects in orthonormal AOs convention

sym(A::AbstractMatrix) = Symmetric(0.5 .* (A + transpose(A)))
@doc raw""" 
    project_tangent_DM(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T},
                            A::Matrix{T}, B::Matrix{T}) where {T<:Real}
For a given point ``(P_d^ᵒ, P_s^ᵒ)`` on the DM manifold (in orthonormal MO convention),
project the couple ``(A,B)`` on the tangent space at ``(P_d^ᵒ, P_s^ᵒ)``.
"""
function project_tangent_DM(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T},
                            A::Matrix{T}, B::Matrix{T}) where {T<:Real}
    Pvᵒ = Symmetric(I - Pdᵒ - Psᵒ)
    Πdᵒ = sym(Pdᵒ*(A-B)*Psᵒ) + 2*sym(Pdᵒ*A*Pvᵒ)
    Πsᵒ = sym(Pdᵒ*(B-A)*Psᵒ) + 2*sym(Psᵒ*B*Pvᵒ)
    Πdᵒ, Πsᵒ
end

@doc raw"""
    gradient_DM_metric(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T}, ζ::State{T}) where {T<:Real}

Gradient of the energy at point ``(P_d^ᵒ, P_s^ᵒ)`` on the DM manifold (in 
orthonormal MO convention).
"""
function gradient_DM_metric(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T}, ζ::State{T}) where {T<:Real}
    @assert(ζ.isortho)
    # ortho -> non-ortho densities
    Fdᵒ, Fsᵒ = Fock_operators(Pdᵒ, Psᵒ, ζ)
    hcat(project_tangent_DM(Pdᵒ, Psᵒ, 2*Fdᵒ, 2*Fsᵒ)...)
end

@doc raw"""
    energy_and_gradient_DM_metric(Pdᵒ, Psᵒ, Sm12, mo_numbers, eri, H, mol)

Energy and gradient of the energy at point ``(P_d^ᵒ, P_s^ᵒ)`` on the DM manifold (in 
orthonormal MO convention).
"""
function ROHF_energy_and_DM_gradient(Pdᵒ, Psᵒ, Sm12, mo_numbers, eri, H, mol)
    Pd, Ps = Symmetric(Sm12*Pdᵒ*Sm12), Symmetric(Sm12*Psᵒ*Sm12)
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    # energy
    E = ROHF_energy(Pd, Ps, Jd, Js, Kd, Ks, H, mol)
    # gradient
    Fdᵒ, Fsᵒ = Fock_operators(Jd, Js, Kd, Ks, H, Sm12)
    ∇E = hcat(project_tangent_DM(Pdᵒ, Psᵒ, 2*Fdᵒ, 2*Fsᵒ)...)
    E, ∇E
end
function ROHF_energy_and_DM_gradient(Pdᵒ::Matrix{T}, Psᵒ::Matrix{T}, ζ::State{T}) where {T<:Real}
    ROHF_energy_and_DM_gradient(Pdᵒ, Psᵒ, ζ.Σ.Sm12, collect(ζ)[1:end-1]...)
end
