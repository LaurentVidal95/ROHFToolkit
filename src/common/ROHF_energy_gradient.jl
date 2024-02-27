@doc raw"""
    assemble_CX_operators(eri::Vector{T}, Pi::AbstractMatrix{T},
                               Pa::AbstractMatrix{T}) where {T<:Real}

Assemble the Coulomb and exchange matrices J and K for doubly occupied
and singly occupied densities, in non-orthogonal AO basis convention.
Arguments are the molecular integrals in a vector shape (reduced by symmetry)
and the densities Pi and Pa.

Links toward PySCF doc:
 - https://pyscf.org/_modules/pyscf/gto/moleintor.html
 - https://pyscf.org/_modules/pyscf/scf/hf.html
"""
function assemble_CX_operators(eri::Vector{T}, Pi::AbstractMatrix{T},
                               Pa::AbstractMatrix{T}) where {T<:Real}
    @assert ( issymmetric(Pi) && issymmetric(Pa) ) "Densities have to be symmetric"
    # TODO setup threading, see pyscf lib.num_threads(n_threads)
    # Compute coulomb and exchange operators
    Ji, Ki = pyscf.scf.hf.dot_eri_dm(eri, Pi, hermi=1)
    Ja, Ka = pyscf.scf.hf.dot_eri_dm(eri, Pa, hermi=1)
    Ji, Ja, Ki, Ka
end

###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                       Energy                         !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
"""
Low level ROHF energy function. Arguments are the
doubly and singly occupied states density and coulomb/exchange operators.
The core hamiltonian as well as the PySCF molecule is also needed.
"""
ROHF_energy(Pi, Pa, Ji, Ja, Ki, Ka, H, mol) =
    tr(H*(2Pi+Pa)) + tr((2Ji-Ki)*(Pi+Pa)) + 1/2*tr((Ja-Ka)*Pa) + mol.energy_nuc()

@doc raw"""
    ROHF_energy(Pi::Matrix{T}, Pa::Matrix{T}, mo_numbers,
                     eri::Vector{T},  H::Matrix{T}, mol)

ROHF energy given only densities, core Hamiltonian and the molecule
as a pyscf object.
"""
function ROHF_energy(Pi::Matrix{T}, Pa::Matrix{T}, mo_numbers,
                     eri::Vector{T},  H::Matrix{T}, mol) where {T<:Real}
    Ji, Ja, Ki, Ka = assemble_CX_operators(eri, Pi, Pa)
    ROHF_energy(Pi, Pa, Ji, Ja, Ki, Ka, H, mol)
end
ROHF_energy(Pi::Matrix, Pa::Matrix, ζ::State) =
    ROHF_energy(Pi, Pa, collect(ζ)[1:end-1]...)

@doc raw"""
    ROHF_energy(C::Matrix{T}, ζ::State{T}) where {T<:Real}

Higher level ROHF energy function, given only a set of MOs and a State.
"""
function ROHF_energy(C::Matrix{T}, ζ::State{T}) where {T<:Real}
    Pi, Pa = densities(C, ζ.Σ.mo_numbers)
    ROHF_energy(Pi, Pa, collect(ζ)[1:end-1]...)
end
function ROHF_energy(ζ::State)
    @assert ζ.isortho
    ROHF_energy(ζ.Σ.Sm12*ζ.Φ, ζ)
end

@doc raw"""
    Fock_operators(Ji, Ja, Ki, Ka, H, Sm12)

Computes the Fock operators Fi and Fa in orthonormal AO basis,
given all necessary matrices:
    - Coulomb and exchange operators Ji, Ja, Ki and Ka
    - The core hamiltonian matrix H
    - The square-root inverse of the overlap ```S^{-1/2}```.
"""
function Fock_operators(Ji, Ja, Ki, Ka, H, Sm12)
    Fi = H .+ 2Ji .- Ki .+ Ja .- 0.5 .* Ka
    Fa =  0.5 .*(H .+ 2Ji .- Ki .+ Ja .- Ka)
    Sm12*Fi*Sm12, Sm12*Fa*Sm12
end

function Fock_operators(Φ, Sm12, mo_numbers, eri, H)
    Pi, Pa = densities(Sm12*Φ, mo_numbers)
    Ji, Ja, Ki, Ka = assemble_CX_operators(eri, Pi, Pa)
    Fock_operators(Ji, Ja, Ki, Ka, H, Sm12)
end

@doc raw"""
    Fock_operators(Φ, ζ::State)

Compact routine to compute the Fock operators Fi and Fa
for a given state in the ROHF manifold, in orthonormal AO basis.
"""
Fock_operators(Φ, ζ::State) = Fock_operators(Φ, ζ.Σ.Sm12, collect(ζ)[1:end-2]...)
function Fock_operators(ζ::State)
    @assert(ζ.isortho)
    Fock_operators(ζ.Φ, ζ)
end
function Fock_operators(Piᵒ, Paᵒ, ζ::State{T}) where {T<:Real}
    _, eri, H = collect(ζ)[1:end-2]
    Sm12 = ζ.Σ.Sm12
    Pi = Symmetric(Sm12*Piᵒ*Sm12); Pa = Symmetric(Sm12*Paᵒ*Sm12)
    Ji, Ja, Ki, Ka = assemble_CX_operators(eri, Pi, Pa)
    Fock_operators(Ji, Ja, Ki, Ka, H, Sm12)
end

function ROHF_gradient_κ(Φ, Fi, Fa, mo_numbers)
    # Extract MO numbers
    Nb, Ni, Na = mo_numbers
    Ne = Nb - (Ni+Na)
    # Compute κ
    Φi, Φa, Φe = split_MOs(Φ, (Nb, Ni, Na); virtuals=true)
    Gx = -Φi'*(Fi-Fa)*Φa
    Gy = -2*Φi'Fi*Φe
    Gz = -2*Φa'Fa*Φe
    return [zeros(Ni,Ni) Gx Gy; -Gx' zeros(Na,Na) Gz; -Gy' -Gz' zeros(Ne,Ne)]
end
function ROHF_gradient(Φ, ζ::State)
    @assert(ζ.isortho)
    @assert(ζ.virtuals)
    # Fock operators in orthonormal AO conventions
    Fi, Fa = Fock_operators(Φ, ζ.Σ.Sm12, collect(ζ)[1:end-2]...)

    # Assemble the kappa matrix of the gradient and return as a TangentVector
    κ = ROHF_gradient_κ(Φ, Fi, Fa, ζ.Σ.mo_numbers)
    TangentVector(κ, ζ)
end
ROHF_gradient(ζ::State) = ROHF_gradient(ζ.Φ, ζ)

@doc raw"""
    ROHF_energy_and_gradient(Φ, ζ::State)

Computes both energy and gradient at point [Φ] on the MO manifold.
"""
function ROHF_energy_and_gradient(Φ, Sm12, mo_numbers, eri, H, mol)
    # Compute Ji, Ja, Ki, Ka
    Pi, Pa = densities(Sm12*Φ, mo_numbers) # Densities in non-orthonormal AOs convention.
    Ji, Ja, Ki, Ka = assemble_CX_operators(eri, Pi, Pa)
    # energy
    E = ROHF_energy(Pi, Pa, Ji, Ja, Ki, Ka, H, mol)
    # gradient
    Fi, Fa = Fock_operators(Ji, Ja, Ki, Ka, H, Sm12)
    κ = ROHF_gradient_κ(Φ, Fi, Fa, mo_numbers)

    # Return energy and gradient
    E, κ
end
function ROHF_energy_and_gradient(Φ, ζ::State)
    @assert(ζ.isortho)
    @assert(ζ.virtuals)
    E, κ = ROHF_energy_and_gradient(Φ, ζ.Σ.Sm12, collect(ζ)[1:end-1]...)
    E, TangentVector(κ, ζ)
end
ROHF_energy_and_gradient(ζ::State; kwargs...) = ROHF_energy_and_gradient(ζ.Φ, ζ)

# Old functions for XC operators
# function tensor_slice(mol::PyObject, i, j, type)
#     shls_slice = nothing
#     n_ao = mol.nao
#     (type=="J") && (shls_slice = (i-1, i-1, j-1, j-1, 0, n_ao-1, 0, n_ao-1)) # Marche pas
#     (type=="K") && (shls_slice = (i-1, i-1, 0, n_ao-1, 0, n_ao-1, j-1, j-1,)) # Marche pas
#     mol.intor("int2e", shls_slice=shls_slice)
# end

# function manual_CX_operators(A::AbstractArray{T}, Pi::AbstractMatrix{T},
#                              Pa::AbstractMatrix{T}) where {T<:Real}
#     Ji = zero(Pi); Ja = zero(Pa); Ki = zero(Pi); Ka = zero(Pa);
#     N = size(Pi,1)
#     for j in 1:N
#         for i in j:N
#             A_J = A[i,j,:,:] # Remplacer par la slice ci dessus
#             A_K = A[i,:,:,j] # Remplacer par la slice ci dessus
#             Ji[i,j] = tr(A_J*Pi); Ji[j,i] = Ji[i,j] # Ji = J(Pi)
#             Ja[i,j] = tr(A_J*Pa); Ja[j,i] = Ja[i,j] # Ja = J(Pa)
#             Ki[i,j] = tr(A_K*Pi); Ki[j,i] = Ki[i,j] # Ki = K(Pi)
#             Ka[i,j] = tr(A_K*Pa); Ka[j,i] = Ka[i,j] # Ka = K(Pa)
#         end
#     end
#     Ji, Ja, Ki, Ka
# end

# """
# test = mol.intor("int2e", shls_slice=(0, 1, 0, 1, 0, 5, 0, 5), aosym="2ij")
# dropdims(test, dims=1) donne A[1,1,:,:]
# """
