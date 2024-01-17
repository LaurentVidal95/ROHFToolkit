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
