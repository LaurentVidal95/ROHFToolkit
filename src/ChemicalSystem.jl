@doc raw""" 
    ChemicalSystem(mol::PyObject, mo_numbers::Tuple(Int64,Int64,Int64),
        overlap_matrix::Matrix{T}, eri::Vector{T}, core_hamiltonian::Matrix{T},
        S12::Matrix{T}, Sm12::Matrix{T}) where {T<:Real}

Class built above a pyscf molecule object that
contains all the data needed for optimization:
  - mol: molecule as a pyscf object, containing data on atoms and number and
    spin of electrons.
  - mo_numbers: tuple (Nb, Nd, Ns), respectively the total number, the number of
    doubly-occupied and the number singly occupied MOs.
  - overlap_matrix: the overlap matrix for the given GTO basis
  - eri: the four index tensor in a compressed format
  - core_hamiltonian: the core hamiltonian matrix
  - S12: square root of the overlap matrix for MO (de)orthonormalization
  - Sm12: inverse square root of the overlap matrix for MO
    (de)orthonormalization
"""
struct ChemicalSystem{T <: Real}
    # Molecule containing the geometry and numbers Nb, Nd, Ns
    mol               ::PyObject
    mo_numbers :: Tuple{Int64,Int64,Int64}
    # Static object to compute energy
    overlap_matrix    ::AbstractMatrix{T}
    eri               ::Vector{T}
    core_hamiltonian  ::AbstractMatrix{T}
    # overlap dependent
    S12               ::AbstractMatrix{T}
    Sm12              ::AbstractMatrix{T}
end

@doc raw"""
Construct a ChemicalSystem for a pyscf molecule object.
"""
function ChemicalSystem(mol::PyObject)
    No, Nd = mol.nelec; Ns = No - Nd; Nb = convert(Int64, mol.nao);
    mo_numbers = (Nb, Nd, Ns)
    # Compute static objects
    S = mol.intor("int1e_ovlp")
    eri = mol.intor("int2e", aosym="s8")
    H = mol.intor("int1e_nuc") + mol.intor("int1e_kin")
    #
    S12 = sqrt(Symmetric(S))
    Sm12 = inv(S12)
    # Assemble chemical system
    Σ = ChemicalSystem{eltype(S)}(mol, mo_numbers, S, eri, H, S12, Sm12)
end

"""
Overload of collect to quickly extract the needed information
"""
Base.collect(Σ::ChemicalSystem) = (Σ.mo_numbers, Σ.eri, Σ.core_hamiltonian,
                                   Σ.mol, Σ.overlap_matrix)
