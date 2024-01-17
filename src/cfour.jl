# Hack to work with CFOUR

function extract_CFOUR_data(CFOUR_file::String)
    @assert isfile(CFOUR_file) "Not a file"
    # extract raw data
    data = readdlm(CFOUR_file)
    multipop(tab, N) = map(x->pop!(tab), 1:N)

    # Extract MO numbers and energy
    Ni, Na, Ne = multipop(data, 3)
    Nb = Ni+Na+Ne
    mo_numbers = (Nb, Ni, Na)
    E = pop!(data)

    # Check that the numbers match
    len_XYZ = Ni*Na + Ni*Ne + Na*Ne
    data_lenght = (len_XYZ + 2*Nb^2)

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
    S = reshape(data, (Nb, Nb))

    @assert issymetric(S)
    
    # Remove external orbitals
    Iₒ = Matrix(I, Nb, Ni+Na)
    ∇E = Φ*A*Iₒ
    Φₒ = Φ*Iₒ
    mo_numbers, Φₒ, E, ∇E
end

"""
Assemble dummy ROHFState to match the code convention
"""
function CASSCFState(mo_numbers, Φ::AbstractArray{T}, E_init) where {T<:Real}
    Nb, Ni, Na = mo_numbers
    # Assemble dumyy ROHFState
    mol = convert(PyObject, nothing)
    S = zeros(T, Nb, Nb)
    H = S; S12=S; Sm12=S
    eri = T[]
    Σ_dummy = ChemicalSystem{eltype(Φ)}(mol, mo_numbers, S, eri, H, S12, Sm12)
    
    guess=:external
    history = reshape([0, E_init, NaN, NaN], 1, 4)

    ROHFState(Φ, Σ_dummy, E_init, false, guess, history)
end

function CASSCF_energy_and_gradient(ζ::ROHFState)
    Φ = ζ.Φ
    open("current_orbitals.txt", "w") do file
        println.(Ref(file), Φ)
    end

    # run CFOUR
    CFOUR_file = run_CFOUR()

    # Extract CFOUR data and return E, ∇E
    _, _, E, ∇E = extract_CFOUR_data(CFOUR_file)
    E, ∇E
end

