# Define ROHF manifold, associated methods and objects


#Structure that only serves to match Optim.jl standards.
#Contains almost no information but is attached to the retraction
#and projection methods.
mutable struct ROHFManifold <: Manifold
    mo_numbers :: Tuple{Int64,Int64,Int64}
end

#
#Molecular orbitals belonging to a specified ROHF manifold.
#
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

The guess is one of the following (see PySCF API doc):
:minao, :atom, :huckel, :hcore, :hcore_pyscf (1e in PySCF doc), :chkfile

The hcore guess of PySCF ("1e") is somehow very bad so it is handled manualy.
See the "core_guess" function bellow.
"""
function init_guess(Σ::ChemicalSystem{T}, M::ROHFManifold, guess::Symbol) where {T<:Real}
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
    (guess == :hcore) && (return core_guess(Σ, M.mo_numbers))
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
function ROHFState(Σ::ChemicalSystem{T}; guess=:minao) where {T<:Real}
    mol = Σ.mol
    # Create Manifold and ChemicalSystem
    No, Nd = mol.nelec; Ns = No - Nd; Nb = convert(Int64, mol.nao);
    M = ROHFManifold((Nb, Nd, Ns))
    # Compute guess
    Φ_init = init_guess(Σ, M, guess)
    E_init = rohf_energy(densities(Φ_init, M.mo_numbers)..., M.mo_numbers, collect(Σ)[1:3]...)
    ROHFState(Φ_init, Σ, M, E_init, false)
end
ROHFState(mol::PyObject; guess=:minao) = ROHFState(ChemicalSystem(mol), guess=guess)
ROHFState(ζ::ROHFState, Φ::Matrix) = ROHFState(Φ, ζ.Σ, ζ.M, ζ.energy, ζ.isortho)

#
# If vec = foot.Φ, ROHFTangentVector is just a ROHFState
#
mutable struct ROHFTangentVector{T<:Real}
    vec::AbstractMatrix{T}
    foot::ROHFState{T}
end

ROHFTangentVector(ζ::ROHFState) = ROHFTangentVector(ζ.Φ, Φ)

function reset_state!(ζ::ROHFState; guess=:minao)
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
    orthonormalize_state!(Ψ.foot; S12)
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


#
#Retraction and vector transports on the ROHF manifold.
#
function retract(M::ROHFManifold, Ψ::ROHFTangentVector{T}) where {T<:Real}
    Nb, Nd, Ns = M.mo_numbers
    No = Nd+Ns
    (Ψd, Ψs), (Φd, Φs) = split_MOs(Ψ)

    # d <-> s rotations
    X = -Φd'Ψs
    W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';

    # occupied <-> virtual
    Ψd_tilde = Ψd .- Φd*Φd'*Ψd .- Φs*Φs'*Ψd;
    Ψs_tilde = Ψs .- Φd*Φd'*Ψs .- Φs*Φs'*Ψs;
    V1,D,V2 = svd(hcat(Ψd_tilde, Ψs_tilde))
    Σ = diagm(D)

    ROHFState(Ψ.foot, (Ψ.foot.Φ*V2*cos(Σ) + V1*sin(Σ))*V2' * exp(W))
end
function retract!(M::ROHFManifold, Ψ::ROHFTangentVector{T}) where {T<:Real}
    Ψ.vec = retract(M, Ψ).Φ
    nothing
end

"""
For (Ψd,Ψs) in R^{Nb×Nd}×R^{Nb×Ns} and y = (Φd,Φs) in the MO manifold
the orthogonal projector on the horizontal tangent space at y is defined by

Π_y(Ψd|Ψs) = ( 1/2*Φs[Φs'Ψd - Ψs'Φd] + Φv(Φv'Ψd) | -1/2*Φd[Ψd'Φs - Φd'Ψs] + Φv(Φv'Ψs) )

If Φ is no the foot of Ψ, may serve as an alternative to transport 
Ψ in the tangent plane to Φ.
"""
function project_tangent(Φ::Matrix{T}, Ψ::Matrix{T}, mo_numbers) where {T<:Real}
    Ψd, Ψs = split_MOs(Ψ, mo_numbers)
    Φd, Φs = split_MOs(Φ, mo_numbers)
    X = 1/2 .* (Ψd'Φs + Φd'Ψs);
    hcat(-Φs*X' + (I - Φd*Φd')*Ψd, -Φd*X + (I - Φs*Φs')*Ψs)
end
project_tangent(M::ROHFManifold, Φ::Matrix, Ψ::Matrix) =
    project_tangent(Φ, Ψ, M.mo_numbers)
function project_tangent!(M::ROHFManifold, Ψ::ROHFTangentVector)
    Ψ.vec = project_tangent(M, Ψ.foot, Ψ.vec)
    nothing
end

"""
Transport a direction p along t*p. Used as default transport for SD and CG
in the linesearch routine on the ROHF manifold.
"""
function transport_vec_along_himself(Ψ::ROHFTangentVector{T}, t::T,
                                     ζ_next::ROHFState{T}) where {T<:Real}
    # Check that the targeted point is in orthonormal AO convention
    @assert (ζ_next.isortho)

    Nb, Nd, Ns = ζ_next.M.mo_numbers
    No = Nd+Ns
    (Ψd, Ψs), (Φd, Φs) = split_MOs(Ψ)
    Φd_next, Φs_next = split_MOs(ζ_next)
    
    # d <-> s rotations
    X = -Φd'*Ψs;
    W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';

    # occupied <-> virtual
    Ψd_tilde = Ψd .- Φd*Φd'*Ψd .- Φs*Φs'*Ψd;
    Ψs_tilde = Ψs .- Φd*Φd'*Ψs .- Φs*Φs'*Ψs;
    V1,D,V2 = svd(hcat(Ψd_tilde, Ψs_tilde))
    Σ = diagm(D)

    τ_p = (-Ψ.foot.Φ*V2*sin(t .*Σ) + V1*cos( t .*Σ))*Σ*V2' * exp(t .* W) + ζ_next.Φ*W
    Ξd, Ξs = split_MOs(τ_p, (Nb,Nd,Ns))
    ROHFTangentVector(hcat(Ξd - Φd_next*Φd_next'Ξd, Ξs - Φs_next*Φs_next'Ξs), ζ_next)
end
