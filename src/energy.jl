# TODO: Change the name of gradient routines...

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
energy(Pd, Ps, Jd, Js, Kd, Ks, H, mol) =
    tr(H*(2Pd+Ps)) + tr((2Jd-Kd)*(Pd+Ps)) + 1/2*tr((Js-Ks)*Ps) + mol.energy_nuc()

"""
Energy given only densities, core Hamiltonian and molecule.
"""
function energy(Pd::Matrix{T}, Ps::Matrix{T}, mo_numbers,
                     eri::Vector{T},  H::Matrix{T}, mol) where {T<:Real}
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    energy(Pd, Ps, Jd, Js, Kd, Ks, H, mol)
end
energy(Pd::Matrix, Ps::Matrix, ζ::ROHFState) =
    energy(Pd, Ps, collect(ζ)[1:end-1]...)

"""
Energy given only MOs and ROHFState.
"""
function energy(Φ::Matrix{T}, ζ::ROHFState{T}) where {T<:Real}  
    Pd, Ps = densities(Φ, ζ.Σ.mo_numbers)
    energy(Pd, Ps, collect(ζ)[1:end-1]...)
end
function energy!(ζ::ROHFState{T}) where {T<:Real}
    Φ = ζ.Φ
    (ζ.isortho) && (Φ = ζ.Σ.Sm12*Φ)
    E = energy(Φ, ζ)
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
"""
function Fock_operators(Jd, Js, Kd, Ks, H, Sm12)
    FdT = H .+ 2Jd .- Kd .+ Js .- 0.5 .* Ks
    FsT = 0.5 .*(H .+ 2Jd .- Kd .+ Js .- Ks)
    Sm12*FdT*Sm12, Sm12*FsT*Sm12
end
"""
Computes the Fock operators Fd and Fs for a given state
in the ROHF manifold, in orthonormal AO basis.
"""
function Fock_operators(Φ, Sm12, mo_numbers, eri, H)
    Pd, Ps = densities(Sm12*Φ, mo_numbers)
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    Fock_operators(Jd, Js, Kd, Ks, H, Sm12)
end
Fock_operators(Φ, ζ::ROHFState) = Fock_operators(Φ, ζ.Σ.Sm12, collect(ζ)[1:end-2]...)
function Fock_operators(ζ::ROHFState)
    @assert(ζ.isortho)
    Fock_operators(ζ.Φ, ζ)
end

### DM formalism
function Fock_operators(PdT, PsT, ζ::ROHFState{T}) where {T<:Real}
    _, eri, H = collect(ζ)[1:end-2]
    Sm12 = ζ.Σ.Sm12
    Pd = Symmetric(Sm12*PdT*Sm12); Ps = Symmetric(Sm12*PsT*Sm12)
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    Fock_operators(Jd, Js, Kd, Ks, H, Sm12)
end


"""
Compute the gradient of the energy in MO formalism for the metric: 
g_y(Ψd, Ψs) = ⟨Ψd,Ψs⟩_{MO}
For a state Φ = (Φd, Φs), the gradient lies in the horizontal tangent space at y:
∇gE_MO(y) = ( Φs[Φs'2(Fd-Fs)Φd] + Φv[4Φv'FdΦd],  -Φd[Φd'2(Fd-Fs)Φs] + Φv[4Φv'FsΦs] )
"""
function ambiant_space_gradient(Φ, Sm12, mo_numbers, eri, H)
    FdT, FsT = Fock_operators(Φ, Sm12, mo_numbers, eri, H)
    ΦdT, ΦsT = split_MOs(Φ, mo_numbers);
    hcat(4*FdT*ΦdT, 4*FsT*ΦsT)    
end
function gradient_MO_metric(Φ, Sm12, mo_numbers, eri, H)
    ∇E = ambiant_space_gradient(Φ, Sm12, mo_numbers, eri, H)
    project_tangent(mo_numbers, Φ, ∇E)
end
function gradient_MO_metric(Φ, ζ::ROHFState)
    @assert(ζ.isortho)
    ∇E = gradient_MO_metric(Φ, ζ.Σ.Sm12, collect(ζ)[1:end-2]...)
    ROHFTangentVector(∇E, ζ)
end

function energy_and_gradient(Φ, Sm12, mo_numbers, eri, H, mol)
    # Compute Jd, Js, Kd, Ks
    Pd, Ps = densities(Sm12*Φ, mo_numbers) # Densities in non-orthonormal AOs convention.
    Jd, Js, Kd, Ks = assemble_CX_operators(eri, Pd, Ps)
    # energy
    E = energy(Pd, Ps, Jd, Js, Kd, Ks, H, mol)
    # gradient
    FdT, FsT = Fock_operators(Jd, Js, Kd, Ks, H, Sm12)
    ΦdT, ΦsT = split_MOs(Φ, mo_numbers);
    ∇E = project_tangent(mo_numbers, Φ, hcat(4*FdT*ΦdT, 4*FsT*ΦsT))
    #
    E, ∇E
end
function energy_and_gradient(Φ, ζ::ROHFState)
    E, ∇E = energy_and_gradient(Φ, ζ.Σ.Sm12, collect(ζ)[1:end-1]...)
    E, ROHFTangentVector(∇E, ζ)
end
function energy_and_gradient(ζ::ROHFState)
    @assert(ζ.isortho)
    energy_and_gradient(ζ.Φ, ζ)
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
#             A_J = A[i,j,:,:] # Remplacer par la slice ci dessus
#             A_K = A[i,:,:,j] # Remplacer par la slice ci dessus
#             Jd[i,j] = tr(A_J*Pd); Jd[j,i] = Jd[i,j] # Jd = J(Pd)
#             Js[i,j] = tr(A_J*Ps); Js[j,i] = Js[i,j] # Js = J(Ps)
#             Kd[i,j] = tr(A_K*Pd); Kd[j,i] = Kd[i,j] # Kd = K(Pd)
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
