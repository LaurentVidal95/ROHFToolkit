using Optim

"""
Molecular orbitals belonging to a specified ROHF manifold.
"""
mutable struct ROHFState{T<:Real}
    Φ::AbstractMatrix{T}
    M::ROHFManifold
    energy::T
end

function ROHFState(mol::PyObject; guess_type="huckel")
    # Create Manifold
    No, Nd = mol.nelec; Ns = No - Nd; Nb = convert(Int64, mol.nao);
    M = ROHFManifold((Nb, Nd, Ns))
    # Compute initial guess
    rohf = pyscf.scf.ROHF(mol);
    rohf.kernel(max_cycle=0, init_guess=guess_type, verbose=0)
    Φ_init = rohf.mo_coeff[:,1:mol.nelectron]
    ROHFState(Φ_init, M, rohf.energy_tot())
end

"""
If vec = foot.Φ, ROHFTangentVector is just a ROHFState
"""
mutable struct ROHFTangentVector{T<:Real}
    vec::AbstractMatrix{T}
    foot::ROHFState{T}
end
ROHFTangentVector(state::ROHFState{T}) = ROHFTangentVector(state.Φ, Φ)
    
mutable struct ROHFManifold <: Manifold
    mo_numbers :: Vector{Int64}
end

# function retract!(M::ROHF_Manifold, Ψ::ROHFTangentVector{T}) where {T<:Real}
#     Nb, Nd, Ns = M.N_bds; No = Nd+Ns
#     Ψd, Ψs = split_MOs(Ψ, N_bds); Φd, Φs = split_MOs(M.foot, N_bds);
    
#     # d <-> s rotations
#     X = -Φd'Ψs
#     W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';
    
#     # occupied <-> virtual
#     Ψd_tilde = Ψd .- Φd*Φd'*Ψd .- Φs*Φs'*Ψd;
#     Ψs_tilde = Ψs .- Φd*Φd'*Ψs .- Φs*Φs'*Ψs;
#     V1,D,V2 = svd(hcat(Ψd_tilde, Ψs_tilde))
#     Σ = diagm(D)
    
#     Ψ .= (Φ*V2*cos(Σ) + V1*sin(Σ))*V2' * exp(W)
# end

# function project_tangent!(M::ROHF_Manifold, Ψ::ROHFTangentVector{T}) where {T<:Real}
#     Φd, Φs = split_MOs(M.foot, N_bds);
#     Ψd, Ψs = split_MOs(Ψ, M.N_bds)
#     X = 1/2 .* (ΨdT'ΦsT + ΦdT'ΨsT); I = diagm(ones(N_bds[1]));
#     Ψ .= -ΦsT*X' + (I - ΦdT*ΦdT')*ΨdT, -ΦdT*X + (I - ΦsT*ΦsT')*ΨsT
# end
