"""
Structure that only serves to match Optim.jl standards.
Contains almost no information but is attached to the retraction
and projection methods.
"""
mutable struct ROHFManifold <: Manifold
    mo_numbers :: Tuple{Int64,Int64,Int64}
end

"""
Molecular orbitals belonging to a specified ROHF manifold.
"""
mutable struct ROHFState{T<:Real}
    Φ::AbstractMatrix{T}
    Σ::ChemicalSystem{T}
    M::ROHFManifold
    #TODO Change to a Energy struct to display units and handle conversions
    energy::T
    isortho::Bool
end
"""
Returns initial guess MOs and energies
The guess is a string following PySCF conventions.
[TODO: LIST ALL PYSCF GUESS]
"""
function init_guess(mol, guess::String)
    pyscf = pyimport("pyscf")
    rohf = pyscf.scf.ROHF(mol)
    rohf.kernel(max_cycle=0, init_guess=guess, verbose=0)
    rohf.mo_coeff[:,1:mol.nelec[1]], rohf.energy_tot()
end

function ROHFState(Σ::ChemicalSystem; guess="huckel")
    mol = Σ.mol
    # Create Manifold and ChemicalSystem
    No, Nd = mol.nelec; Ns = No - Nd; Nb = convert(Int64, mol.nao);
    M = ROHFManifold((Nb, Nd, Ns))
    Φ_init, E_init = init_guess(mol, guess)
    ROHFState(Φ_init, Σ, M, E_init, false)
end
ROHFState(mol::PyObject; guess="huckel") = ROHFState(ChemicalSystem(mol), guess=guess)
ROHFState(ζ::ROHFState, Φ::Matrix) = ROHFState(Φ, ζ.Σ, ζ.M, ζ.energy, ζ.isortho)    

"""
If vec = foot.Φ, ROHFTangentVector is just a ROHFState
"""
mutable struct ROHFTangentVector{T<:Real}
    vec::AbstractMatrix{T}
    foot::ROHFState{T}
end

ROHFTangentVector(ζ::ROHFState) = ROHFTangentVector(ζ.Φ, Φ)

function reset_state!(ζ::ROHFState; guess="huckel")
    Φ_init, E = init_guess(ζ.Σ.mol, guess)
    ζ.Φ = Φ_init; ζ.energy = E; ζ.isortho=false;
    nothing
end

function orthonormalize_state!(ζ::ROHFState; S12=sqrt(Symmetric(ζ.Σ.overlap_matrix)))
    # If MOs are already orthonormal do nothing
    (ζ.isortho) && (@info "The state is already orthonomal")
    if !(ζ.isortho)
        ζ.Φ = S12*ζ.Φ
        ζ.isortho = true
    end
    nothing
end
orthonormalize_state!(Ψ::ROHFTangentVector; S12=sqrt(Symmetric(Ψ.foot.Σ.overlap_matrix)))=
    orthonormalize_state!(Ψ.foot, S12=S12)
function deorthonormalize_state!(ζ::ROHFState;
                                 Sm12=inv(sqrt(Symmetric(ζ.Σ.overlap_matrix))))
    !(ζ.isortho) && (@info "The state is already non-orthonomal")
    if (ζ.isortho)
        ζ.Φ = Sm12*ζ.Φ
        ζ.isortho = false
    end
    nothing
end
function deorthonormalize_state!(Ψ::ROHFTangentVector;
                                 Sm12=inv(sqrt(Symmetric(Ψ.foot.Σ.overlap_matrix))))
    deorthonormalize_state!(Ψ.foot, S12=S12)
end

"""
Gathers all informations except the MOs Φ
"""
Base.collect(ζ::ROHFState) = (ζ.M.mo_numbers, collect(ζ.Σ)...)
