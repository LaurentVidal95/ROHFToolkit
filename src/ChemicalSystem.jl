# TODO: exploit low memory storage of the four indexed tensor of PySCF.
# For now the tensor is stored as a full Array.

"""
Class containing all infos of the system:
 - mol = molecule as a pyscf object, containing data
   on atoms and number and spin of electrons.
 - the overlap matrix
 - the four index tensor
 - the core hamiltonian matrix
 - MOs in non orthogonal AO basis
 - E_rohf: the ROHF energy.
"""
struct ChemicalSystem{T <: Real}
    # Molecule containing the geometry and numbers Nb, Nd, Ns
    mol               ::PyObject
    ## Static object to compute energy
    overlap_matrix    ::AbstractMatrix{T}
    eri               ::Vector{T}
    # A :: AbstractArray{T}
    core_hamiltonian  ::AbstractMatrix{T}
end

"""
    Construct a chemical system directly from a file
"""
function ChemicalSystem(mol::PyObject)
    pyscf = pyimport("pyscf")
    ## Compute static objects
    S = mol.intor("int1e_ovlp")
    eri = mol.intor("int2e", aosym="s8")
    # A = mol.intor("int2e")
    # A = pyscf.ao2mo.restore(1, A, size(S,1))
    H = mol.intor("int1e_nuc") + mol.intor("int1e_kin")
    ## Assemble chemical system
    Σ = ChemicalSystem{eltype(S)}(mol, S, eri, H)
    # Σ = ChemicalSystem{eltype(S)}(mol, S, A, H)
end

Base.collect(Σ::ChemicalSystem) = (Σ.eri, Σ.core_hamiltonian,
                                   Σ.mol, Σ.overlap_matrix)
