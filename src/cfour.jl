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
    mo_numbers = (Nb, Ni, Na)
    E = popfirst!(data)

    ∇E_cfour = reshape(multipop(data, Nb^2), Nb, Nb)
    @assert(norm(∇E_cfour' + ∇E_cfour) < 1e-10)
    # Extract orbitals and overlap
    Φ_cfour = reshape(multipop(data, Nb^2), Nb, Nb)
    S_cfour = reshape(multipop(data, Nb^2), Nb, Nb)

    # Extract core Hamiltonian, Fock matrices and tensor
    H_cfour = reshape(multipop(data, Nb^2), Nb, Nb)
    Fi_cfour = reshape(multipop(data, Nb^2), Nb, Nb)
    Fa_cfour = reshape(multipop(data, Nb^2), Nb, Nb)
    Q_cfour = reshape(multipop(data, Nb^2), Nb, Nb)
    T_cfour = reshape(multipop(data, Nb^4), Nb, Nb, Nb, Nb)

    # Sanity checks
    @assert norm(Φ_cfour'S_cfour*Φ_cfour - I) < 1e-7
    @assert isempty(data)
    @assert norm(S_cfour-S_cfour') < 1e-10

    # Remove external orbitals and assemble Stiefel gradient
    (;mo_numbers, mo_coeffs=Φ_cfour, overlap=S_cfour, energy=E, gradient=∇E_cfour)
end

# Extract gradient blocs X, Y and Z as vector XYZ
# len_XYZ = Ni*Na + Ni*Ne + Na*Ne
# XYZ = multipop(data, len_XYZ)
# function reshape_XYZ(XYZ, N1, N2, N3)
#     X = reshape(multipop(XYZ, N1*N2), N1, N2)
#     Y = reshape(multipop(XYZ, N1*N3), N1, N3)
#     Z = reshape(XYZ, N2, N3)
#     X,Y,Z
# end
# X, Y, Z = reshape_XYZ(XYZ, Ni, Na, Ne)
# ∇E_cfour = -2 .* [zeros(Ni, Ni) X Y; -X' zeros(Na, Na) Z; -Y' -Z' zeros(Ne, Ne)]
# ∇E_cfour=(X,Y,Z)

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
function CASSCF_energy_and_gradient(ζ::State; CFOUR_ex="xcasscf", verbose=true)
    @assert ζ.isortho

    # De-orthonormalize Φ_tot
    Φ = ζ.Σ.Sm12*ζ.Φ
    open("current_orbitals.txt", "w") do file
        println.(Ref(file), Φ)
    end

    # run and extract CFOUR data
    _ = run_CFOUR(CFOUR_ex; verbose)
    data = extract_CFOUR_data("energy_gradient.txt")
    Φ_ortho = ζ.Σ.S12*data.mo_coeffs

    # Compute gradient in AMO parametrization from kappa parametrization
    Nb, Ni, Na = data.mo_numbers
    E = data.energy
    ∇E = TangentVector(Φ_ortho*data.gradient, ζ)
    E, ∇E
end
"""
The standard Backtracking linesearch uses independantly the energy and gradient
functional. This CASSCF_energy is a simple wrapper around the CASSCF_energy_and_gradient
that returns the energy only.
"""
function CASSCF_energy(ζ::State; CFOUR_ex="xcasscf", verbose=true)
    E, ∇E = CASSCF_energy_and_gradient(ζ; CFOUR_ex, verbose)
    E
end
function CASSCF_gradient(ζ::State; CFOUR_ex="xcasscf", verbose=true)
    E, ∇E = CASSCF_energy_and_gradient(ζ; CFOUR_ex, verbose)
    ∇E
end


#################################### TESTS

function energy_landscape(ζ::State, dir::TangentVector;
                          N_step=100,
                          max_step=1e-5)
    Φ = ζ.Φ
    E_landscape = []
    steps = LinRange(-max_step, max_step, N_step)
    progress = Progress(N_step, desc="Computing energy landscape")
    for t in steps
        ζ_next = retract_AMO(ζ, TangentVector(t*dir, ζ); type=:exp)
        E_next, _ = CASSCF_energy_and_gradient(ζ_next)
        push!(E_landscape, E_next)
        next!(progress)
        writedlm("E_landscape.txt", E_landscape)
    end
    E_landscape, steps
end


function random_direction(x::State; target_norm=1., return_B=false)
    B = ROHFToolkit.rand_mmo_matrix(x.Σ.mo_numbers)
    B = B .* (target_norm/norm(B))
    dir_vec = x.Φ*B
    return_B && return TangentVector(dir_vec, x), B
    TangentVector(dir_vec, x)
end

"""
Gradient test by finite difference.
The fg function produces the ROHF energy and gradient or the CASSCF one.
"""
function test_gradient(ζ::State, t; fg=ROHF_energy_and_gradient)
    Nb, Ni, Na = ζ.Σ.mo_numbers

    # Assemble Φ with virtual
    @assert ζ.isortho
    @assert norm(ζ.Φ'ζ.Φ - I) < 1e-10
    Φ = ζ.Φ

    # Random direction in horizontal tangent space
    p = random_direction(ζ)
    @assert test_tangent(p) < 1e-8 # Test that p belongs to the tangent space at ζ

    # Test gradient by finite difference
    Φ_next = retract_AMO(Φ, t*p; type=:exp)
    @assert test_MOs(Φ_next, (Nb, Ni, Na)) < 1e-10  # Test that Φ_next is a point on the AMO manifold

    E, ∇E = fg(ζ)
    @assert test_tangent(∇E) < 1e-8

    approx = (ROHF_energy(ζ.Σ.Sm12*Φ_next, ζ) - energy(ζ.Σ.Sm12*ζ.Φ, ζ))/t
    expected = tr(∇E'p)
    approx/expected, p
end

"""
Old manual test.
"""
# function test_gradient_CFOUR(data, t, mol; ∇E=nothing)
#     Nb, Ni, Na = data.mo_numbers

#     # Orbitals
#     Φ_cfour = data.mo_coeffs
#     Pd, Ps = densities(Φ_cfour, data.mo_numbers)
#     Jd, Js, Kd, Ks = ROHFToolkit.manual_CX_operators(data.tensor, Pd, Ps)
#     E = energy(Pd, Ps, Jd, Js, Kd, Ks, data.core_hamiltonian, mol)
#     @assert norm(data.energy -  E) < 1e-6

#     @assert norm(Φ_cfour'data.overlap*Φ_cfour - I) < 1e-10
#     Φₒ = Φ_cfour*Matrix(I, Nb, Ni+Na)
#     @assert norm(Φₒ'data.overlap*Φₒ - I) < 1e-10
#     S12 = sqrt(Symmetric(data.overlap))
#     Φₒ_ortho = S12*Φₒ
#     Φ_cfour_ortho = S12*Φ_cfour

#     # Living in orthonormal world
#     B = rand_mmo_matrix(data.mo_numbers)     # Random direction in horizontal tangent space
#     p = Φ_cfour_ortho*B*Matrix(I, Nb, Ni+Na)
#     ∇E_cfour = isnothing(∇E) ? data.gradient : ∇E
#     @assert norm(project_tangent(data.mo_numbers, Φₒ_ortho, p) -p) < 1e-10
#     @assert norm(project_tangent(data.mo_numbers, Φₒ_ortho, ∇E_cfour) - ∇E_cfour) < 1e-10

#     Φₒ_ortho_next = Φ_cfour_ortho*exp(t*B)*Matrix(I, Nb, Ni+Na)
#     @assert norm(Φₒ_ortho_next'Φₒ_ortho_next - I) < 1e-10

#     # Manual energy orthonormal -> non orthonormal
#     Φₒ_next = inv(S12)*Φₒ_ortho_next
#     Pd, Ps = densities(Φₒ_next, data.mo_numbers)
#     Jd, Js, Kd, Ks = ROHFToolkit.manual_CX_operators(data.tensor, Pd, Ps)
#     E_next = energy(Pd, Ps, Jd, Js, Kd, Ks, data.core_hamiltonian, mol)

#     approx = (E_next - E)/t
#     expected = tr(∇E_cfour'p)
#     (;test = norm(approx - expected), direction=p)
# end
