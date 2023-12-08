# Useful tool to write gradient and projectors in a compact way

sym(A::AbstractArray) = Symmetric( 1/2 .*(A .+ transpose(A)) )

@doc raw"""
    split_MOs(Φ, mo_numbers)

From given matrix ```Φ=[Φ_d|Φ_s]`` containing both groups of MOs,
return the separate matrices ``Φ_d`` and ``Φ_s``.
"""
function split_MOs(Φ, mo_numbers)
    Nb,Nd,Ns = mo_numbers
    Φd = Φ[:,1:Nd]; Φs = Φ[:,Nd+1:Nd+Ns]
    Φd, Φs
end
split_MOs(ζ::ROHFState) = split_MOs(ζ.Φ, ζ.Σ.mo_numbers)
split_MOs(Ψ::ROHFTangentVector) = split_MOs(Ψ.vec, Ψ.base.Σ.mo_numbers), split_MOs(Ψ.base)

@doc raw"""
Compute densities ``P_d = Φ Φ^{T}``, ``P_s = Φ Φ^{T}``
from MOs in orthonormal AO basis.
"""
function densities(Φᵒ, mo_numbers)
    # Extract needed integers
    Nb,Nd,Ns = mo_numbers
    Φdᵒ = Φᵒ[:,1:Nd]; Φsᵒ = Φᵒ[:,Nd+1:Nd+Ns]
    Φdᵒ*Φdᵒ', Φsᵒ*Φsᵒ'
end
densities(ζ::ROHFState) = densities(ζ.Φ, ζ.Σ.mo_numbers)

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
@doc raw"""
    vec_to_mat(XYZ, mo_numbers)

Reverse operation of ``mat_to_vec``.
The mo_numbers are needed to recover the proper dimensions
of X, Y and Z as three individual matrices.
"""
function vec_to_mat(XYZ, mo_numbers)
    Nb,Nd,Ns = mo_numbers
    Nn = Nd*Ns + Nb*Nd + Nb*Ns
    # reshape
    X = reshape(XYZ[1:Nd*Ns],(Nd,Ns))
    Y = reshape(XYZ[Nd*Ns+1:Nd*Ns+Nb*Nd],(Nb,Nd))
    Z = reshape(XYZ[Nd*Ns+Nb*Nd+1:Nn],(Nb,Ns))
    [X,Y,Z]
end

@doc raw"""
     test_MOs(Φ::Matrix{T}, mo_numbers) where {T<:Real}

Test if the give MOs are in the MO manifold.
"""
function test_MOs(Φ::Matrix{T}, mo_numbers) where {T<:Real}
    Nb,Nd,Ns = mo_numbers
    Pd, Ps = densities(Φ, mo_numbers)

    test = norm(Pd*Pd - Pd)
    test += norm(Ps*Ps - Ps)
    test += norm(Pd*Ps)
    test += tr(Pd) - Nd
    test -= tr(Ps) - Ns

    test
end
test_MOs(ζ::ROHFState) = test_MOs(ζ.Φ, ζ.Σ.mo_numbers)

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
rand_unitary_matrix(ζ::ROHFState) = rand_unitary_matrix(ζ.Σ.mo_numbers)


## The following was only used to test routines using geodesics.
## See the end of ``src/common/MO_manifold_tools.jl``.
# """
# Orthonormalize vector b with the n first columns of A
# """
# function gram_schmidt(A::AbstractArray{T}, b::AbstractVector{T},n::Integer) where T
#     k = size(A,2)  #number of orthogonal vectors
#     dot_prods = zeros(T,k)  
    
#     for (i,a_i) in enumerate(eachcol(A[:,1:n]))
#         dot_prods[i] = dot(a_i,b)
#         axpy!(-dot_prods[i],a_i,b)
#     end
#     nrm = norm(b)
#     b *= one(T)/nrm
#     b
# end

# """
# Uses the preceding routine to obtain a full set of orthonormalized
# orbitals from occupied ones. MOs are given in orthonormal AO convention.
# """
# function generate_virtual_MOs(Φdᵒ::AbstractArray{T},Φsᵒ::AbstractArray{T},
#                            N_bds) where T
#     #Initialize new orbs, fill virtuals randomly
#     Nb,Nd,Ns = N_bds; No = Nd + Ns
#     Φᵒ = zeros(T,Nb,Nb)
#     Φᵒ[:,1:Nd+Ns] = hcat(Φdᵒ,Φsᵒ)
#     virtuals = rand(T,Nb,Nb-(Nd+Ns))

#     for (i,v) in enumerate(eachcol(virtuals))
#         ortho_v = gram_schmidt(Φᵒ, v, No+(i-1))
#         Φᵒ[:,No+i] = ortho_v
#     end

#     Φᵒ[:,No+1:end]
# end
