# Define ROHF manifold, associated methods and objects


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
Returns initial guess MOs and energies in non-orthonormal convention
The guess is one of the following (see PySCF API doc)
["minao", "atom", "huckel", "hcore", "1e", "chkfile".]
"""
function init_guess(Σ::ChemicalSystem{T}, M::ROHFManifold, guess::String) where {T<:Real}
    # Define PySCF rohf object
    pyscf = pyimport("pyscf")
    rohf = pyscf.scf.ROHF(Σ.mol)
    
    # Dictionary of all PySCF init guess
    @assert guess ∈ ["hcore", "minao", "atom", "huckel", "1e", "chkfile"]  "Guess not handled"
    init_guesses = Dict("minao"   => rohf.init_guess_by_minao,
                        "atom"    => rohf.init_guess_by_atom,
                        "huckel"  => rohf.init_guess_by_huckel,
                        "1e"      => rohf.init_guess_by_1e,
                        "chkfile" => rohf.init_guess_by_chkfile)

    # Handle core guess manualy
    (guess == "hcore") && (return core_guess(Σ, M.mo_numbers))

    # Other guesses via PySCF
    P_ortho = Symmetric(init_guesses[guess]()[1,:,:])
    No = sum(M.mo_numbers[2:3])
    Φ_ortho = eigen(-P_ortho).vectors[:,1:No]

    # Deorthonormalize
    inv(sqrt(Symmetric(Σ.overlap_matrix))) * Φ_ortho
end

function core_guess(Σ::ChemicalSystem{T}, mo_numbers) where {T<:Real}
    No = sum(mo_numbers[2:3])
    F = eigen(Symmetric(Σ.core_hamiltonian), Symmetric(Σ.overlap_matrix))
    normalize_col(col) = col ./ sqrt(col'*Σ.overlap_matrix*col)
    hcat(normalize_col.(eachcol(F.vectors[:,1:No]))...)
end

"""
All ROHFSate structure constructors
"""
function ROHFState(Σ::ChemicalSystem{T}; guess="minao") where {T<:Real}
    mol = Σ.mol
    # Create Manifold and ChemicalSystem
    No, Nd = mol.nelec; Ns = No - Nd; Nb = convert(Int64, mol.nao);
    M = ROHFManifold((Nb, Nd, Ns))
    # Compute guess
    Φ_init = init_guess(Σ, M, guess)
    E_init = rohf_energy(densities(Φ_init, M.mo_numbers)..., M.mo_numbers, collect(Σ)[1:3]...)
    ROHFState(Φ_init, Σ, M, E_init, false)
end
ROHFState(mol::PyObject; guess="minao") = ROHFState(ChemicalSystem(mol), guess=guess)
ROHFState(ζ::ROHFState, Φ::Matrix) = ROHFState(Φ, ζ.Σ, ζ.M, ζ.energy, ζ.isortho)

"""
If vec = foot.Φ, ROHFTangentVector is just a ROHFState
"""
mutable struct ROHFTangentVector{T<:Real}
    vec::AbstractMatrix{T}
    foot::ROHFState{T}
end

ROHFTangentVector(ζ::ROHFState) = ROHFTangentVector(ζ.Φ, Φ)

function reset_state!(ζ::ROHFState; guess="minao")
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

"""
Gathers all informations except the MOs Φ
"""
Base.collect(ζ::ROHFState) = (ζ.M.mo_numbers, collect(ζ.Σ)...)
