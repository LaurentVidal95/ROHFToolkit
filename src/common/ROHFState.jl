# Define ROHF manifold, associated methods and objects

import Base.+, Base.-, Base.*, Base.adjoint, Base.vec
import LinearAlgebra.norm

#
#Molecular orbitals belonging to a specified ROHF manifold.
#
mutable struct ROHFState{T<:Real}
    Φ       ::AbstractMatrix{T}
    Σ       ::ChemicalSystem{T}
    energy  ::T
    #
    isortho ::Bool
    guess   ::Symbol
    # All the history of minimizing precedure is contained in ζ
    # so that it can be updated by OptimKit
    history ::Matrix{T}
end

"""
Returns initial guess MOs and energies in non-orthonormal convention

The guess is one of the following (see PySCF API doc):
:minao, :atom, :huckel, :hcore, :hcore_pyscf (1e in PySCF doc), :chkfile

The hcore guess of PySCF ("1e") is somehow very bad so it is handled manualy.
See the "core_guess" function bellow.
"""
function init_guess(Σ::ChemicalSystem{T}, guess::Symbol) where {T<:Real}
    # Define PySCF rohf object
    rohf = pyscf.scf.ROHF(Σ.mol)

    # Dictionary of all PySCF init guess
    @assert guess ∈ [:hcore, :minao, :atom, :huckel, :chkfile, :hcore_pyscf] "Guess not handled"
    init_guesses = Dict(:minao            => rohf.init_guess_by_minao,
                        :atom             => rohf.init_guess_by_atom,
                        :huckel           => rohf.init_guess_by_huckel,
                        :hcore_pyscf      => rohf.init_guess_by_1e,
                        :chkfile          => rohf.init_guess_by_chkfile)
    # Handle core guess manualy because the PySCF core guess is somehow very bad..
    (guess == :hcore) && (return core_guess(Σ))
    # Other guesses via PySCF
    P_ortho = Symmetric(init_guesses[guess]()[1,:,:])
    No = sum(Σ.mo_numbers[2:3])
    Φ_ortho = eigen(-P_ortho).vectors[:,1:No]

    # Deorthonormalize
    inv(sqrt(Symmetric(Σ.overlap_matrix))) * Φ_ortho
end

function core_guess(Σ::ChemicalSystem{T}) where {T<:Real}
    No = sum(Σ.mo_numbers[2:3])
    F = eigen(Symmetric(Σ.core_hamiltonian), Symmetric(Σ.overlap_matrix))
    normalize_col(col) = col ./ sqrt(col'*Σ.overlap_matrix*col)
    hcat(normalize_col.(eachcol(F.vectors[:,1:No]))...)
end

"""
All ROHFSate structure constructors
"""
function ROHFState(Σ::ChemicalSystem{T}; guess=:minao) where {T<:Real}
    # Compute guess
    Φ_init = init_guess(Σ, guess)
    E_init = energy(densities(Φ_init, Σ.mo_numbers)..., collect(Σ)[1:4]...)
    history = reshape([0, E_init, NaN, NaN], 1, 4)
    ROHFState(Φ_init, Σ, E_init, false, guess, history)
end
ROHFState(mol::PyObject; guess=:minao) = ROHFState(ChemicalSystem(mol); guess)
ROHFState(ζ::ROHFState, Φ::Matrix) = ROHFState(Φ, ζ.Σ, ζ.energy, ζ.isortho, ζ.guess, ζ.history)

#
# If vec = base.Φ, ROHFTangentVector is just a ROHFState
#
struct ROHFTangentVector{T<:Real}
    vec::AbstractMatrix{T}
    base::ROHFState{T}
end

ROHFTangentVector(ζ::ROHFState) = ROHFTangentVector(ζ.Φ, Φ)
(+)(A::Matrix, X::ROHFTangentVector) = (+)(A, X.vec)
(+)(X::ROHFTangentVector, A::Matrix) = (+)(X.vec, A)
(*)(λ::Real, X::ROHFTangentVector) = (*)(λ, X.vec)
(*)(A::Adjoint{Float64, Matrix{Float64}}, X::ROHFTangentVector) = (*)(A, X.vec)
(-)(X::ROHFTangentVector) = -X.vec
norm(X::ROHFTangentVector) = norm(X.vec)
(adjoint)(X::ROHFTangentVector) = X.vec'
(vec)(X::ROHFTangentVector) = vec(X.vec)

function reset_state!(ζ::ROHFState; guess=:minao)
    Φ_init = init_guess(ζ.Σ, guess)
    ζ.Φ = Φ_init; ζ.isortho=false; energy!(ζ);
    history = reshape([0, ζ.energy, NaN, NaN], 1, 4)
    ζ.history = history
    nothing
end

function orthonormalize_state!(ζ::ROHFState)
    # If MOs are already orthonormal do nothing
    (ζ.isortho) && (@info "The state is already orthonomal")
    if !(ζ.isortho)
        ζ.Φ = ζ.Σ.S12*ζ.Φ
        ζ.isortho = true
    end
    nothing
end
orthonormalize_state!(Ψ::ROHFTangentVector) = orthonormalize_state!(Ψ.base)
function deorthonormalize_state!(ζ::ROHFState)
    !(ζ.isortho) && (@info "The state is already non-orthonomal")
    if (ζ.isortho)
        ζ.Φ = ζ.Σ.Sm12*ζ.Φ
        ζ.isortho = false
    end
    nothing
end

"""
Gathers all informations except the MOs Φ
"""
Base.collect(ζ::ROHFState) = collect(ζ.Σ)
