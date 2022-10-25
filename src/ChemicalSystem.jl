# Class containing all info on the system:
#  - mol = molecule as a pyscf object, containing data
#    on atoms and number and spin of electrons.
#  - the overlap matrix
#  - the four index tensor
#  - the core hamiltonian matrix
#  - MOs in non orthogonal AO basis
#  - E_rohf: the ROHF energy.
struct ChemicalSystem{T <: Real}
    # Molecule containing the geometry and numbers Nb, Nd, Ns
    mol               ::PyObject
    mo_numbers :: Tuple{Int64,Int64,Int64}
    # Static object to compute energy
    overlap_matrix    ::AbstractMatrix{T}
    eri               ::Vector{T}
    core_hamiltonian  ::AbstractMatrix{T}
    # overlap dependant
    S12               ::AbstractMatrix{T}
    Sm12              ::AbstractMatrix{T}
end

"""
Construct a chemical system directly from a file
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

Base.collect(Σ::ChemicalSystem) = (Σ.mo_numbers, Σ.eri, Σ.core_hamiltonian,
                                   Σ.mol, Σ.overlap_matrix)
