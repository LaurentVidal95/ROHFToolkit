# Hack to work with CFOUR

function extract_CFOUR_data(CFOUR_file::String)
    @assert isfile(CFOUR_file) "Not a file"
    # extract raw data
    data = readdlm(CFOUR_file)
    #
    Ni, Na, Ne = data[1:3]
    Nb = Ni+Na+Ne
    mo_numbers = (Nb, Ni, Na)
    #
    E = data[4]
    # Extract gradient blocs X, Y and Z as vector XYZ
    len_XYZ = Ni*Na + Ni*Ne + Na*Ne
    XYZ = data[5:5+len_XYZ]
    X, Y, Z = vec_to_mat(XYZ, mo_numbers)
    A = [zeros(Ni, Ni) X Y; -X' zeros(Na, Na) Z; -Y' -Z' zeros(Ne, Ne)]
    # Extract orbitals
    Φ = reshape(data[6+len_XYZ:end], Nb, Nb)

    # Reshape gradient
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
    
    nothing
end
# function CASSCF_energy_and_gradient(CFOUR_dir)
#     @assert isdir(CFOUR_dir)
#     # Extract data

#     # 
# end
