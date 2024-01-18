# Define ROHF manifold, associated methods and objects

import Base.+, Base.-, Base.*, Base.adjoint, Base.vec
import LinearAlgebra.norm

@doc raw"""
    ROHFState(Φ::Matrix, Σ::ChemicalSystem, energy::Real, isortho::Bool,
                         guess::String, history::Matrix)

Molecular orbitals belonging to a specified ROHF manifold.
"""
mutable struct ROHFState{T<:Real}
    Φ        ::AbstractMatrix{T}
    Σ        ::ChemicalSystem{T}
    energy   ::T
    #
    isortho  ::Bool
    guess    ::Symbol
    virtuals ::Bool # Select the All MOs or occupied MOs manifold
    # All the history of minimizing precedure is contained in ζ
    # so that it can be updated by OptimKit
    history  ::Matrix{T}
end

"""
Gathers all informations except the MOs Φ.
"""
Base.collect(ζ::ROHFState) = collect(ζ.Σ)

@doc raw"""
    generate_molden(ζ::ROHFState, filename::String)

Save the MOs contained in the state ``ζ`` in a molden file
for visualization.
"""
function generate_molden(ζ::ROHFState, filename::String)
    pyscf.tools.molden.from_mo(ζ.Σ.mol, filename, ζ.Φ)
    nothing
end

@doc raw"""
    init_guess(Σ::ChemicalSystem{T}, guess::Symbol) where {T<:Real}

Returns initial guess MOs and energies in non-orthonormal convention.
The guess is one of the following (see PySCF API doc):
    - :minao
    - :atom
    - :huckel
    - :hcore
    - :hcore_pyscf (1e in PySCF doc).
    - :chkfile
The hcore guess of PySCF ("1e") is somehow very bad so it is handled manualy.
See the "core_guess" function bellow.
"""
function init_guess(Σ::ChemicalSystem{T}, guess::Symbol; virtuals=true) where {T<:Real}
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
    (guess == :hcore) && (return core_guess(Σ; virtuals))
    # Other guesses via PySCF
    P_ortho = Symmetric(init_guesses[guess]()[1,:,:])
    No = sum(Σ.mo_numbers[2:3])
    guess_MOs = eigen(-P_ortho).vectors
    Φ_ortho = virtuals ? all_guess_MOs : guess_MOs[:,1:No]

    # Deorthonormalize
    inv(sqrt(Symmetric(Σ.overlap_matrix))) * Φ_ortho
end
@doc raw"""
    core_guess(Σ::ChemicalSystem{T}) where {T<:Real}

Initial guess obtained by diagonalizing the core hamiltonian matrix,
and using the Aufbau principle.
"""
function core_guess(Σ::ChemicalSystem{T}; virtuals=true) where {T<:Real}
    No = sum(Σ.mo_numbers[2:3])
    F = eigen(Symmetric(Σ.core_hamiltonian), Symmetric(Σ.overlap_matrix))
    normalize_col(col) = col ./ sqrt(col'*Σ.overlap_matrix*col)
    guess_MOs = hcat(normalize_col.(eachcol(F.vectors))...)
    Φ = virtuals ? guess_MOs : guess_MOs[:,1:No]
end

"""
All ROHFSate structure constructors
"""
function ROHFState(Σ::ChemicalSystem{T}; guess=:minao, virtuals=true) where {T<:Real}
    # Compute guess
    Φ_init = init_guess(Σ, guess; virtuals)
    E_init = energy(densities(Φ_init, Σ.mo_numbers)..., collect(Σ)[1:4]...)
    history = reshape([0, E_init, NaN, NaN], 1, 4)
    ROHFState(Φ_init, Σ, E_init, false, guess, virtuals, history)
end
ROHFState(mol::PyObject; guess=:minao, virtuals=true) = ROHFState(ChemicalSystem(mol); guess, virtuals)
ROHFState(ζ::ROHFState, Φ::Matrix) = ROHFState(Φ, ζ.Σ, ζ.energy, ζ.isortho, ζ.guess, ζ.virtuals, ζ.history)

@doc raw"""
     ROHFTangentVector(vec::AbstractMatrix{T}, base::ROHFState{T})

The base vector is a point on the MO manifold, and vec is a vector in
the tangent space the base to the MO manifold.
Mainly serve to match the conventions of the OptimKit optimization library.
Note that if vec = base.Φ, the ROHFTangentVector is just a ROHFState.
"""
struct ROHFTangentVector{T<:Real}
    vec::AbstractMatrix{T}
    base::ROHFState{T}
end

"""
Overloading basic operations for ROHFTangentVectors.
"""
ROHFTangentVector(ζ::ROHFState) = ROHFTangentVector(ζ.Φ, Φ)
(+)(A::Matrix, X::ROHFTangentVector) = (+)(A, X.vec)
(+)(X::ROHFTangentVector, A::Matrix) = (+)(X.vec, A)
(*)(λ::Real, X::ROHFTangentVector) = (*)(λ, X.vec)
(*)(A::Adjoint{Float64, Matrix{Float64}}, X::ROHFTangentVector) = (*)(A, X.vec)
(-)(X::ROHFTangentVector) = -X.vec
norm(X::ROHFTangentVector) = norm(X.vec)
(adjoint)(X::ROHFTangentVector) = X.vec'
(vec)(X::ROHFTangentVector) = vec(X.vec)

@doc raw"""
    reset_state!(ζ::ROHFState; guess=:minao)


"""
function reset_state!(ζ::ROHFState; guess=:minao, virtuals=true)
    Φ_init = init_guess(ζ.Σ, guess; virtuals)
    ζ.Φ = Φ_init; ζ.isortho=false; energy!(ζ);
    history = reshape([0, ζ.energy, NaN, NaN], 1, 4)
    ζ.virtuals = virtuals
    ζ.history = history
    nothing
end

@doc raw"""
    orthonormalize_state!(ζ::ROHFState)

Use the square root of the overlap matrix to orthonormalize the
MOs of a given state.
"""
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

@doc raw"""
    deorthonormalize_state!(ζ::ROHFState)

Use the inverse square root of the overlap matrix to deorthonormalize the
MOs of a given state.
"""
function deorthonormalize_state!(ζ::ROHFState)
    !(ζ.isortho) && (@info "The state is already non-orthonomal")
    if (ζ.isortho)
        ζ.Φ = ζ.Σ.Sm12*ζ.Φ
        ζ.isortho = false
    end
    nothing
end
