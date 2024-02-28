# Useful tool to write gradient and projectors in a compact way

sym(A::AbstractArray) = Symmetric( 1/2 .*(A .+ transpose(A)) )
asym(A::AbstractArray) = 1/2 .* (A .- transpose(A))

@doc raw"""
    split_MOs(Φ, mo_numbers)

From given matrix ```Φ=[Φ_d|Φ_s]`` containing both groups of MOs,
return the separate matrices ``Φ_d`` and ``Φ_s``.
"""
function split_MOs(Φ, mo_numbers; virtuals=false)
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
Compute densities ``P_d = Φ Φ^{T}``, ``P_s = Φ Φ^{T}``
from MOs in orthonormal AO basis.
"""
function densities(Φᵒ, mo_numbers)
    # Extract needed integers
    Nb,Nd,Ns = mo_numbers
    Φdᵒ = Φᵒ[:,1:Nd]; Φsᵒ = Φᵒ[:,Nd+1:Nd+Ns]
    Φdᵒ*Φdᵒ', Φsᵒ*Φsᵒ'
end
densities(ζ::State) = densities(ζ.Φ, ζ.Σ.mo_numbers)

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

@doc raw"""
     is_point(Φ::Matrix{T}, mo_numbers) where {T<:Real}

Test if the give MOs are a point in the AMO manifold.
"""
function is_point(Φ::Matrix{T}, mo_numbers; tol=1e-8) where {T<:Real}
    Nb,Nd,Ns = mo_numbers
    Pd, Ps = densities(Φ, mo_numbers)

    test = norm(Pd*Pd - Pd)
    test += norm(Ps*Ps - Ps)
    test += norm(Pd*Ps)
    test += tr(Pd) - Nd
    test -= tr(Ps) - Ns

    return test < tol
end
is_point(ζ::State; tol=1e-8) = is_point(ζ.Φ, ζ.Σ.mo_numbers; tol)

"""
Test is the given tangent vector belongs to the tangent space at X.base.
"""
function is_tangent(X::TangentVector; tol=1e-8, return_value=false)
    Nb, Ni, Na = X.base.Σ.mo_numbers
    Ne = Nb - (Ni+Na)
    No = Ni+Na

    κ = X.kappa

    # check that the diag blocs of κ are zero
    test = zero(eltype(X.base.Φ))
    test += norm(κ[1:Ni, 1:Ni])
    test += norm(κ[Ni+1:No, Ni+1:No])
    test += norm(κ[No+1:Nb, No+1:Nb])

    # test that κ is antisymmetric
    test += norm(κ + κ')
    (return_value) && (return (test<tol), test)

    return test < tol
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
