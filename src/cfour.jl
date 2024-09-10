"""
Small wrapper to run CFOUR in verbose or not verbose
"""
function run_CFOUR(CFOUR_ex; verbose=false)
    run_cmd = verbose ? run : cmd->read(cmd, String)
    res = run_cmd(`$(CFOUR_ex)`)
end

"""
Small wrapper to remind you to remove the previous init files
"""
function CFOUR_init(CFOUR_ex)
    @warn "Have you removed the old file ?"
    run_CFOUR(CFOUR_ex)
    extract_CFOUR_data("energy_gradient.txt")
end

"""
Parse the data in the file provide by CFOUR. The ouput is
    • mo_numbers: number of basis functions, internal and active orbitals (Nb, Ni, Na)
    • Φ_cfour: full set of MOs (i, a and e) in a non-orthonormal convention
    • Φₒ: Same than Φ_cfour without the virtuals
    • the overlap matrix for the CFOUR AO basis
    • the energy at point Φ_cfour
    • the gradient at point Φ_cfour
    • the core hamiltonian and 4-index integral tensor
"""
function extract_CFOUR_data(CFOUR_file::String)
    @assert isfile(CFOUR_file) "Not a file"
    multipop(tab, N) = [popfirst!(tab) for x in tab[1:N]]

    # Extract raw data
    data = vec(readdlm(CFOUR_file))

    # Extract MO numbers and energy
    Ni, Na, Ne = Int.(multipop(data, 3))
    Nb = Ni+Na+Ne
    N_rot = Ni*Na + Ni*Ne + Na*Ne
    mo_numbers = (Nb, Ni, Na)
    E = popfirst!(data)

    ∇E_cfour = reshape(multipop(data, Nb^2), Nb, Nb)
    
    @assert(norm(∇E_cfour' + ∇E_cfour) < 1e-10)
    # Extract orbitals and overlap
    Φ_cfour = reshape(multipop(data, Nb^2), Nb, Nb)
    S_cfour = reshape(multipop(data, Nb^2), Nb, Nb)

    P∇E_cfour = reshape(multipop(data, Nb^2), Nb, Nb)
    hess_diag_matrix = reshape(multipop(data, Nb^2), Nb, Nb)

    # Sanity checks
    @assert isempty(data)
    @assert norm(Φ_cfour'S_cfour*Φ_cfour - I) < 1e-7
    @assert norm(S_cfour-S_cfour') < 1e-10

    # Remove external orbitals and assemble Stiefel gradient
    (;mo_numbers, mo_coeffs=Φ_cfour, overlap=S_cfour,
     energy=E, gradient=∇E_cfour, prec_gradient=P∇E_cfour, hess_diag_matrix)

end

"""
Assemble dummy State to match the code convention
"""
function CASSCFState(mo_numbers, Φ::AbstractArray{T}, S::AbstractArray{T},
                     E_init; virtuals=true) where {T<:Real}
    Nb, Ni, Na = mo_numbers

    mol = convert(PyObject, nothing)
    S12 = sqrt(Symmetric(S))
    Sm12 = inv(S12)
    # Moving on the Manifold can be done without H and the electron repulsion integrals.
    H = zero(S12)
    eri = T[]

    # Assemble dummy chemical system.
    Σ_dummy = ChemicalSystem{eltype(Φ)}(mol, mo_numbers, S, eri, H, S12, Sm12)
    guess=:CFOUR_CASSCF
    history = reshape([0, E_init, NaN, NaN, NaN], 1, 5)

    State(Φ, Σ_dummy, E_init, false, guess, true, history)
end

"""
Call CFOUR to compute the gradient and energies to the current set
of orbitals
"""
function CASSCF_energy_and_gradient(ζ::State; CFOUR_ex="xcasscf", verbose=true, tol_ci)
    @assert ζ.isortho

    # De-orthonormalize Φ_tot
    Φ = ζ.Σ.Sm12*ζ.Φ
    open("current_orbitals.txt", "w") do file
        println(file, tol_ci)
        println.(Ref(file), Φ)
    end

    # run and extract CFOUR data
    _ = run_CFOUR(CFOUR_ex; verbose)
    data = extract_CFOUR_data("energy_gradient.txt")

    # Compute gradient in AMO parametrization from kappa parametrization
    Nb, Ni, Na = data.mo_numbers
    E = data.energy
    ∇E = TangentVector(data.gradient, ζ)
    E, ∇E
end
"""
The standard Backtracking linesearch uses independantly the energy and gradient
functional. This CASSCF_energy is a simple wrapper around the CASSCF_energy_and_gradient
that returns the energy only.
"""
function CASSCF_energy(ζ::State; CFOUR_ex="xcasscf", verbose=true, tol_ci=nothing)
    E, ∇E = CASSCF_energy_and_gradient(ζ; CFOUR_ex, verbose, tol_ci)
    E
end
function CASSCF_gradient(ζ::State; CFOUR_ex="xcasscf", verbose=true, tol_ci=nothing)
    E, ∇E = CASSCF_energy_and_gradient(ζ; CFOUR_ex, verbose, tol_ci)
    ∇E
end
function CASSCF_preconditioner(η::TangentVector; tol=1e-6)
    @assert isfile("energy_gradient.txt")
    data = extract_CFOUR_data("energy_gradient.txt")    
    P_inv = data.hess_diag_matrix
    Pη = map(zip(η.kappa, P_inv)) do (x,y)
        y = iszero(y) ? 1. : abs(y)
        x/y
    end
    TangentVector(Pη, η.base)
end
function CASSCF_fixed_diag_preconditioner(η::TangentVector; tol=0.2)
    @assert isfile("energy_gradient.txt")
    data = extract_CFOUR_data("energy_gradient.txt")   
    current_diag = data.hess_diag_matrix
    if !isfile("casscf_diag.txt")
        writedlm("casscf_diag.txt", current_diag)
    end
    diag = readdlm("casscf_diag.txt")
    if norm(diag - current_diag) > tol
        diag = current_diag
        writedlm("casscf_diag.txt", current_diag)
        writedlm("RESTART_LBFGS", rand())
    end
    Pη = map(zip(η.kappa, diag)) do (x,y)
        y = iszero(y) ? 1. : abs(y)
        x/y
    end
    TangentVector(Pη, η.base)
end
