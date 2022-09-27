"""
Assemble the Coulomb and exchange matrices J and K for doubly occupied
and singly occupied densities, in non-orthogonal AO basis convention.
Arguments are the molecular integrals in a vector shape (reduced by symmetry)
and the densities Pd and Ps.
"""
function assemble_CX_operators(eri::Vector{T}, Pd::AbstractMatrix{T},
                               Ps::AbstractMatrix{T}) where {T<:Real}
    @assert ( issymmetric(Pd) && issymmetric(Ps) ) "Densities have to be symmetric"
    # TODO setup threading, see pyscf lib.num_threads(n_threads)
    # Compute coulomb and exchange operators
    Jd, Kd = pyscf.scf.hf.dot_eri_dm(eri, Pd, hermi=1)
    Js, Ks = pyscf.scf.hf.dot_eri_dm(eri, Ps, hermi=1)
    Jd, Js, Kd, Ks
end

###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                       Energy                         !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
"""
Energy given doubly and singly occupied states density and
coulomb/exchange operators.
The core hamiltonian as well as the PySCF molecule is also needed.
"""
rohf_energy(Pd, Ps, Jd, Js, Kd, Ks, H, mol) =
    tr(H*(2Pd+Ps)) + tr((2Jd-Kd)*(Pd+Ps)) + 1/2*tr((Js-Ks)*Ps) + mol.energy_nuc()

"""
Energy given only densities, core Hamiltonian and molecule.
"""
function rohf_energy(Pd::Matrix{T}, Ps::Matrix{T}, mo_numbers,
                     eri::Vector{T},  H::Matrix{T}, mol) where {T<:Real}
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    rohf_energy(Pd, Ps, Jd, Js, Kd, Ks, H, mol)
end
rohf_energy(Pd::Matrix, Ps::Matrix, ζ::ROHFState) =
    rohf_energy(Pd, Ps, collect(ζ)[1:end-1]...)

"""
Energy given only MOs and ROHFState.
"""
function rohf_energy(Φ::Matrix{T}, ζ::ROHFState{T}) where {T<:Real}
    Pd, Ps = densities(Φ, ζ.M.mo_numbers)
    rohf_energy(Pd, Ps, collect(ζ)[1:end-1]...)
end
function rohf_energy!(ζ::ROHFState{T}) where {T<:Real}
    @assert (!ζ.isortho) "In orthonormal convention you must provide Sm12"
    E = rohf_energy(ζ.Φ, ζ)
    ζ.energy = E
    E
end
function rohf_energy!(ζ::ROHFState{T}, Sm12) where {T<:Real}
    @assert (ζ.isortho) "In non-orthonormal convention no need for Sm12"
    E = rohf_energy(Sm12*ζ.Φ, ζ)
    ζ.energy = E
    E
end

###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                   Energy gradients                   !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
"""
Computes the Fock operators Fd and Fs given in orthonormal AO basis,
given all necessary matrices:
    - Coulomb and exchange operators Jd, Js, Kd and Ks
    - The core hamiltonian matrix H
    - The square-root inverse of the overlap S^{-1/2}
    - The number of total, resp. doubly and singly occupied AOs.
"""
function compute_Fock_operators(Jd, Js, Kd, Ks, H, Sm12, mo_numbers)
    FdT = H .+ 2Jd .- Kd .+ Js .- 0.5 .* Ks
    FsT = 0.5 .*(H .+ 2Jd .- Kd .+ Js .- Ks)
    Sm12*FdT*Sm12, Sm12*FsT*Sm12
end

"""
Computes the Fock operators Fd and Fs for a given state
in the ROHF manifold, in orthonormal AO basis.
"""
function compute_Fock_operators(Φ, Sm12, mo_numbers, eri, H)
    Pd, Ps = densities(Sm12*Φ, mo_numbers)
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    compute_Fock_operators(Jd, Js, Kd, Ks, H, Sm12, mo_numbers)
end
function compute_Fock_operators(Φ, Sm12, ζ::ROHFState)
    @assert(ζ.isortho)
    compute_Fock_operators(Φ, Sm12, collect(ζ)[1:end-2]...)
end

"""
Compute the gradient of the energy in MO formalism for the metric: 
g_y(Ψd, Ψs) = ⟨Ψd,Ψs⟩_{MO}
For a state Φ = (Φd, Φs), the gradient lies in the horizontal tangent space at y:
∇gE_MO(y) = ( Φs[Φs'2(Fd-Fs)Φd] + Φv[4Φv'FdΦd],  -Φd[Φd'2(Fd-Fs)Φs] + Φv[4Φv'FsΦs] )
"""
function grad_E_MO_metric(Φ, Sm12, mo_numbers, eri, H)
    FdT, FsT = compute_Fock_operators(Φ, Sm12, mo_numbers, eri, H)
    ΦdT, ΦsT = split_MOs(Φ, mo_numbers);
    ∇E = hcat(4*FdT*ΦdT, 4*FsT*ΦsT)
    project_tangent(Φ, ∇E, mo_numbers)
end
function grad_E_MO_metric(Φ, Sm12, ζ::ROHFState)
    @assert(ζ.isortho)
    ∇E = grad_E_MO_metric(Φ, Sm12, collect(ζ)[1:end-2]...)
    ROHFTangentVector(∇E, ζ)
end

function rohf_energy_and_gradient(Φ, Sm12, mo_numbers, eri, H, mol)
    # Compute Jd, Js, Kd, Ks
    Pd, Ps = densities(Sm12*Φ, mo_numbers) # Densities in non-orthonormal AOs convention.
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    # energy
    E = rohf_energy(Pd, Ps, Jd, Js, Kd, Ks, H, mol)
    # gradient
    FdT, FsT = compute_Fock_operators(Jd, Js, Kd, Ks, H, Sm12, mo_numbers)
    ΦdT, ΦsT = split_MOs(Φ, mo_numbers);
    ∇E = project_tangent(Φ, hcat(4*FdT*ΦdT, 4*FsT*ΦsT), mo_numbers)
    #
    E, ∇E
end
function rohf_energy_and_gradient(Φ, Sm12, ζ::ROHFState)
    @assert(ζ.isortho)
    E, ∇E = rohf_energy_and_gradient(Φ, Sm12, collect(ζ)[1:end-1]...)
    E, ROHFTangentVector(∇E, ζ)
end

# function tensor_slice(mol::PyObject, i, j, type)
#     shls_slice = nothing
#     n_ao = mol.nao
#     (type=="J") && (shls_slice = (i-1, i-1, j-1, j-1, 0, n_ao-1, 0, n_ao-1)) # Marche pas
#     (type=="K") && (shls_slice = (i-1, i-1, 0, n_ao-1, 0, n_ao-1, j-1, j-1,)) # Marche pas
#     mol.intor("int2e", shls_slice=shls_slice)
# end

# function assemble_CX_operators(A::AbstractArray{T}, Pd::AbstractMatrix{T},
#                                Ps::AbstractMatrix{T}) where {T<:Real}
#     Jd = zero(Pd); Js = zero(Ps); Kd = zero(Pd); Ks = zero(Ps);
#     N = size(Pd,1)
#     for j in 1:N
#         for i in j:N
# 	    A_J = A[i,j,:,:] # Remplacer par la slice ci dessus
#             A_K = A[i,:,:,j] # Remplacer par la slice ci dessus
# 	    Jd[i,j] = tr(A_J*Pd); Jd[j,i] = Jd[i,j] # Jd = J(Pd)
# 	    Js[i,j] = tr(A_J*Ps); Js[j,i] = Js[i,j] # Js = J(Ps)
# 	    Kd[i,j] = tr(A_K*Pd); Kd[j,i] = Kd[i,j] # Kd = K(Pd)
#             Ks[i,j] = tr(A_K*Ps); Ks[j,i] = Ks[i,j] # Ks = K(Ps)
#         end
#     end
#     Jd, Js, Kd, Ks
# end

# """
# test = mol.intor("int2e", shls_slice=(0, 1, 0, 1, 0, 5, 0, 5), aosym="2ij")
# dropdims(test, dims=1) donne A[1,1,:,:]
# """

# Liens doc PySCF
# https://pyscf.org/_modules/pyscf/gto/moleintor.html
# https://pyscf.org/_modules/pyscf/scf/hf.html
