# Hack to work with CFOUR
function run_CFOUR(CFOUR_ex)
    res = read(`$(CFOUR_ex)`, String)
end

function CFOUR_init(CFOUR_ex)
    @warn "Have you removed the old file ?"
    run_CFOUR(CFOUR_ex)
    mo_numbers, Φₒ, S, E, ∇E = extract_CFOUR_data("energy_gradient.txt")
end

function extract_CFOUR_data(CFOUR_file::String)
    @assert isfile(CFOUR_file) "Not a file"
    # extract raw data
    data = vec(readdlm(CFOUR_file))
    multipop(tab, N) = [popfirst!(tab) for x in tab[1:N]]

    # Extract MO numbers and energy
    @show Ni, Na, Ne = Int.(multipop(data, 3))
    Nb = Ni+Na+Ne
    mo_numbers = (Nb, Ni, Na)
    E = popfirst!(data)

    # Check that the numbers match
    len_XYZ = Ni*Na + Ni*Ne + Na*Ne
    data_length = (len_XYZ + 2*Nb^2)

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

    # Reshape gradient and orbitals
    A = [zeros(Ni, Ni) X Y; -X' zeros(Na, Na) Z; -Y' -Z' zeros(Ne, Ne)]
    Φ = reshape(multipop(data, Nb^2), Nb, Nb)
    S = reshape(multipop(data, Nb^2), (Nb, Nb))

    @assert isempty(data)
    @assert norm(S-S') < 1e-10

    # Remove external orbitals
    Iₒ = Matrix(I, Nb, Ni+Na)
    ∇E = Φ*A*Iₒ
    Φₒ = Φ*Iₒ
    mo_numbers, Φₒ, S, E, ∇E
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

    ROHFState(Φ, Σ_dummy, E_init, false, guess, history)
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
    _, _, _, E, ∇E = extract_CFOUR_data("energy_gradient.txt")
    @show E
    E, ROHFTangentVector(∇E, ζ)
end

function energy_lanscape_along_gradient(ζ::ROHFState, ∇E::ROHFTangentVector; N_points=100)
    Φ = ζ.Φ
    E_landscape = []
    for (i,t) in enumerate(LinRange(0, 1, N_points))
        @show i
        Φ_next = retract(ζ, -t*∇E, Φ)
        E_next, _ = CASSCF_energy_and_gradient(ROHFState(ζ, Φ_next))
        push!(E_landscape, E_next)
        writedlm("E_lanscape", E_landscape)
    end
    E_lanscape
end
