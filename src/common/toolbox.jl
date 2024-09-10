sym(A::AbstractArray) = Symmetric( 1/2 .*(A .+ transpose(A)) )
asym(A::AbstractArray) = 1/2 .* (A .- transpose(A))

@doc raw"""
    split_MOs(Φ, mo_numbers)

From given matrix ```Φ=[Φ_d|Φ_s]`` containing both groups of MOs,
return the separate matrices ``Φ_d`` and ``Φ_s``.
"""
function split_MOs(Φ, mo_numbers; virtuals=true)
    Nb,Nd,Ns = mo_numbers
    Φd = Φ[:,1:Nd]; Φs = Φ[:,Nd+1:Nd+Ns]; Φv = Φ[:, Nd+Ns+1:end]
    virtuals && (return Φd, Φs, Φv)
    Φd, Φs
end
split_MOs(ζ::State) = split_MOs(ζ.Φ, ζ.Σ.mo_numbers; ζ.virtuals)
function split_MOs(Ψ::TangentVector)
    ζ=Ψ.base
    split_MOs(Ψ.vec, ζ.Σ.mo_numbers; ζ.virtuals), split_MOs(ζ; ζ.virtuals)
    error("Adapt to new TangentVector convention")
end

@doc raw"""
     mat_to_vec(X,Y,Z)

Concatenate all columns of given matrices X, Y and Z into a single
vector. Used to define preconditioning system with LinearMaps.jl.
"""
function mat_to_vec(X,Y,Z)
    XYZ = Float64[]
    for mat in (X,Y,Z)
        for col in eachcol(mat)
            XYZ = vcat(XYZ, col)
        end
    end
    XYZ
end
function reshape_XYZ(XYZ, N1, N2, N3)
    X = reshape(XYZ[1:N1*N2], N1, N2)
    Y = reshape(XYZ[N1*N2+1:N1*N2+N1*N3], N1, N3)
    Z = reshape(XYZ[N1*N2+N1*N3+1:end], N2, N3)
    X,Y,Z
end
function vec_to_κ(XYZ, Ni, Na, Ne)
    X, Y, Z = reshape_XYZ(XYZ, Ni, Na, Ne)
    [zeros(Ni,Ni) X Y; -X' zeros(Na,Na) Z; -Y' -Z' zeros(Ne,Ne)]
end

@doc raw"""
    rand_unitary_matrix(mo_numbers)

Generate a random unitary matrix with proper dimensions.
"""
function rand_unitary_matrix(mo_numbers)
    Nb, Nd, Ns = mo_numbers
    No = Nd+Ns
    A = rand(No, No)
    exp((1/2)*(A-A'))
end
rand_unitary_matrix(ζ::State) = rand_unitary_matrix(ζ.Σ.mo_numbers)

## The following was only used to test routines using geodesics.
## See the end of ``src/common/MO_manifold_tools.jl``.
"""
    Orthonormalize vector b with the n first columns of A
"""
function gram_schmidt(A::AbstractArray{T}, b::AbstractVector{T},n::Integer) where T
    k = size(A,2)  #number of orthogonal vectors
    dot_prods = zeros(T,k)

    for (i,a_i) in enumerate(eachcol(A[:,1:n]))
        dot_prods[i] = dot(a_i,b)
        axpy!(-dot_prods[i],a_i,b)
    end
    nrm = norm(b)
    b *= one(T)/nrm
    b
end

"""
    Uses the preceding routine to obtain a full set of orthonormalized orbitals from occupied ones.
    MOs are given in orthonormal basis.
"""
function generate_virtual_MOs_T(ΦdT::AbstractArray{T},ΦsT::AbstractArray{T},
                           N_bds) where T
    #Initialize new orbs, fill virtuals randomly
    Nb,Nd,Ns = N_bds; No = Nd + Ns
    ΦT = zeros(T,Nb,Nb)
    ΦT[:,1:Nd+Ns] = hcat(ΦdT,ΦsT)
    virtuals = rand(T,Nb,Nb-(Nd+Ns))

    for (i,v) in enumerate(eachcol(virtuals))
        ortho_v = gram_schmidt(ΦT, v, No+(i-1))
        ΦT[:,No+i] = ortho_v
    end

    ΦT[:,No+1:end]
end

@doc raw"""
    generate_molden(ζ::State, filename::String)

Save the MOs contained in the state ``ζ`` in a molden file
for visualization.
"""
function generate_molden(ζ::State, filename::String)
    pyscf.tools.molden.from_mo(ζ.Σ.mol, filename, ζ.Φ)
    nothing
end
