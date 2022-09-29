"""
    Useful tool, compact wayw to write gradient and projectors.
"""
sym(A::AbstractArray) = Symmetric( 1/2 .*(A .+ transpose(A)) )

function split_MOs(Φ, N_bds)
    Nb,Nd,Ns = N_bds
    Φd = Φ[:,1:Nd]; Φs = Φ[:,Nd+1:Nd+Ns]
    Φd, Φs
end
split_MOs(ζ::ROHFState) = split_MOs(ζ.Φ, ζ.M.mo_numbers)
function split_MOs(Ψ::ROHFTangentVector)
    mo_numbers = Ψ.base.M.mo_numbers
    Φ = Ψ.base.Φ
    split_MOs(Ψ.vec, mo_numbers), split_MOs(Φ, mo_numbers)
end

"""
    Compute densities PdT = ΦT*Id*ΦT', PsT = ΦT*Is*ΦT'
    from MOs in orthonormal AO basis.
"""
function densities(ΦT, N_bds)
    # Extract needed integers
    Nb,Nd,Ns = N_bds
    ΦdT = ΦT[:,1:Nd]; ΦsT = ΦT[:,Nd+1:Nd+Ns]
    ΦdT*ΦdT', ΦsT*ΦsT'
end
densities(ζ::ROHFState) = densities(ζ.Φ, ζ.M.mo_numbers)

"""
Concatenate all columns of given matrices X, Y and Z into a single
vector. Used to define preconditioning system with LinearMaps
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
"""
Reverse operation of mat_to_vec. mo_numbers are needed to recover the proper dimensions
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

"""
    Test if new MOs are admissible solutions MAYBE REMOVE THAT !
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
