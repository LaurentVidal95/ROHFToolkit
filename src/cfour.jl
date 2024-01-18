# Hack to work with CFOUR
function run_CFOUR(CFOUR_ex)
    res = read(`$(CFOUR_ex)`, String)
end

function CFOUR_init(CFOUR_ex)
    @warn "Have you removed the old file ?"
    run_CFOUR(CFOUR_ex)
    extract_CFOUR_data("energy_gradient.txt")
end

function extract_CFOUR_data(CFOUR_file::String)
    @assert isfile(CFOUR_file) "Not a file"
    # extract raw data
    data = vec(readdlm(CFOUR_file))
    multipop(tab, N) = [popfirst!(tab) for x in tab[1:N]]

    # Extract MO numbers and energy
    Ni, Na, Ne = Int.(multipop(data, 3))
    Nb = Ni+Na+Ne
    mo_numbers = (Nb, Ni, Na)
    E = popfirst!(data)

    # Check that the numbers match
    len_XYZ = Ni*Na + Ni*Ne + Na*Ne
    data_length = (2*len_XYZ + 2*Nb^2)

    @assert length(data)==data_length "wrong data length"

    # Extract gradient blocs X, Y and Z as vector XYZ
    XYZ = multipop(data, len_XYZ)
    function reshape_XYZ(XYZ, N1, N2, N3)
        X = reshape(multipop(XYZ, N1*N2), N1, N2)
        Y = reshape(multipop(XYZ, N1*N3), N1, N3)
        Z = reshape(XYZ, N2, N3)
        X,Y,Z
    end
    X, Y, Z = reshape_XYZ(XYZ, Ni, Na, Ne)
    ∇E_cfour = -2 .* [zeros(Ni, Ni) X Y; -X' zeros(Na, Na) Z; -Y' -Z' zeros(Ne, Ne)]
 
    # Extract K matrix bloc and reshape
    XYZ = multipop(data, len_XYZ)
    X, Y, Z = reshape_XYZ(XYZ, Ni, Na, Ne)
    K = [zeros(Ni, Ni) X Y; -X' zeros(Na, Na) Z; -Y' -Z' zeros(Ne, Ne)]

    # Extract orbitals and overlap
    Φ = reshape(multipop(data, Nb^2), Nb, Nb)
    S = reshape(multipop(data, Nb^2), (Nb, Nb))

    @show norm(Φ'S*Φ - I)
    @assert isempty(data)
    @assert norm(S-S') < 1e-10

    # Remove external orbitals and assemble Stiefel gradient
    Iₒ = Matrix(I, Nb, Ni+Na)
    foo = Φ'*S*∇E_cfour
    X = foo[1:Ni, Ni+1:Ni+Na]
    Y = foo[1:Ni, Ni+Na+1:end]
    Z = foo[Ni+1:Ni+Na, Ni+Na+1:end]
    K = [zeros(Ni, Ni) X Y; -X' zeros(Na, Na) Z; -Y' -Z' zeros(Ne, Ne)]
    @show norm(K + K')
    @show size(K)
    @show K
    Φ_ortho = sqrt(Symmetric(S))*Φ
    ∇E = Φ_ortho*K*Iₒ
    # ∇E = inv(sqrt(Symmetric(S)))*∇E_cfour*inv(sqrt(Symmetric(S)))*Iₒ
    Φₒ = Φ_ortho*Iₒ
    (;mo_numbers, Φ_ortho, Φₒ, overlap=S, energy=E, gradient=∇E)
end

"""
Assemble dummy ROHFState to match the code convention
"""
function CASSCFState(mo_numbers, Φ::AbstractArray{T}, S::AbstractArray{T}, E_init) where {T<:Real}
    Nb, Ni, Na = mo_numbers
    # Assemble dumyy ROHFState
    mol = convert(PyObject, nothing)
    S12 = sqrt(Symmetric(S))
    Sm12 = inv(S12)
    H = S
    eri = T[]
    Σ_dummy = ChemicalSystem{eltype(Φ)}(mol, mo_numbers, S, eri, H, S12, Sm12)

    guess=:external
    history = reshape([0, E_init, NaN, NaN], 1, 4)

    ROHFState(Φ, Σ_dummy, E_init, false, guess, false, history)
end

function CASSCF_energy_and_gradient(ζ::ROHFState; CFOUR_ex="xcasscf")
    @assert ζ.isortho
    Φe = generate_virtual_MOs_T(split_MOs(ζ)..., ζ.Σ.mo_numbers)
    Φ_tot = hcat(ζ.Φ, Φe)

    # De-orthonormalize Φ_tot
    Φ_tot = ζ.Σ.Sm12*Φ_tot
    open("current_orbitals.txt", "w") do file
        println.(Ref(file), Φ_tot)
    end

    # run and extract CFOUR data
    _ = run_CFOUR(CFOUR_ex)
    data = extract_CFOUR_data("energy_gradient.txt")
    E = data.energy
    ∇E = data.gradient
    E, ROHFTangentVector(∇E, ζ)
end

function energy_landscape(ζ::ROHFState, dir::ROHFTangentVector;
                          N_step=100,
                          max_step=1e-5)
    Φ = ζ.Φ
    E_landscape = []
    steps = LinRange(-max_step, max_step, N_step)
    progress = Progress(N_step, desc="Computing energy landscape")
    for t in steps
        Φ_next = retract(ζ, t*dir, Φ)
        E_next, _ = CASSCF_energy_and_gradient(ROHFState(ζ, Φ_next))
        push!(E_landscape, E_next)
        next!(progress)
        writedlm("E_landscape.txt", E_landscape)
    end
    E_landscape, steps
end

function rand_mmo_matrix(ζ)
    Nb, Ni, Na = ζ.Σ.mo_numbers
    Ne = Nb - (Na+Ni)
    X = rand(Ni, Na)
    Y = rand(Ni, Ne)
    Z = rand(Na, Ne)
    [zeros(Ni, Ni) X Y; -X' zeros(Na, Na) Z; -Y' -Z' zeros(Ne, Ne)]
end

function test_gradient(ζ::ROHFState, t; ∇E_init=nothing)
    Nb, Ni, Na = ζ.Σ.mo_numbers
    
    # Assemble Φ with virtual
    @assert ζ.isortho
    Φe = generate_virtual_MOs_T(split_MOs(ζ)..., ζ.Σ.mo_numbers)
    Φ_tot = hcat(ζ.Φ, Φe)

    # Random direction in horizontal tangent space
    B = rand_mmo_matrix(ζ)
    p = Φ_tot*B*Matrix(I, Nb, Ni+Na)
    @assert norm(project_tangent(ζ, ζ.Φ, p) - p) < 1e-10

    # Test by finite difference
    Φ_next = Φ_tot*exp(t*B)*Matrix(I, Nb, Ni+Na)
    @assert test_MOs(Φ_next, (11,4,1)) < 1e-10
    
    ∇E = isnothing(∇E_init) ? gradient_MO_metric(ζ.Φ, ζ) : ∇E_init
    @assert norm(project_tangent(ζ, ζ.Φ, ∇E.vec) - ∇E.vec) < 1e-10

    approx = (energy(ζ.Σ.Sm12*Φ_next, ζ) - energy(ζ.Σ.Sm12*ζ.Φ, ζ))/t
    expected = tr(∇E'p)
    approx/expected, p
end

# data = ROHFToolkit.extract_CFOUR_data(file)
# x_init = ROHFToolkit.CASSCFState(data.mo_numbers, data.Φₒ, data.overlap, data.energy)
# ROHFToolkit.orthonormalize_state!(x_init)
function test_gradient_CFOUR(ζ::ROHFState, t; ∇E_init=nothing)
    Nb, Ni, Na = ζ.Σ.mo_numbers
    
    # Assemble Φ with virtual
    @assert ζ.isortho
    Φe = generate_virtual_MOs_T(split_MOs(ζ)..., ζ.Σ.mo_numbers)
    Φ_tot = hcat(ζ.Φ, Φe)

    # Random direction in horizontal tangent space
    B = rand_mmo_matrix(ζ)
    p = Φ_tot*B*Matrix(I, Nb, Ni+Na)
    @assert norm(project_tangent(ζ, ζ.Φ, p) - p) < 1e-10

    # Test by finite difference
    Φ_next = Φ_tot*exp(t*B)*Matrix(I, Nb, Ni+Na)
    @assert norm(Φ_next'Φ_next - I) < 1e-10
    @assert test_MOs(Φ_next, (11,4,1)) < 1e-10
    ζ_next = ROHFState(Φ_next, ζ)
    
    ∇E = isnothing(∇E_init) ? gradient_MO_metric(ζ.Φ, ζ) : ∇E_init
    @assert norm(project_tangent(ζ, ζ.Φ, ∇E) - ∇E) < 1e-10
    
    E_next,_ = CASSCF_energy_and_gradient(ζ_next)
    
    approx = (E_next - E)/t
    expected = tr(∇E'p)
    approx/expected, p
end
