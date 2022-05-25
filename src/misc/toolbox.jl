# TODO : clean old routines. Do some renaming

###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                  General operations                  !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###

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
    mo_numbers = Ψ.foot.M.mo_numbers
    Φ = Ψ.foot.Φ
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

"""
Used for random guesses
"""
function rand_unitary_matrix(N)
    M = rand(N,N)
    exp((M .- M') ./ 2)
end


######## OLD
"""                                                                                 
   Concatenate into a single vector (of size Nb^2) a couple of matrices 
   in the ROHF manifold M
"""                                                                                 
function vectorize_couple_ROHF_manifold(ud, us, N_bds)
    Nb = N_bds[1]; Ntot=2*Nb^2;
    vec=zeros(Ntot,1)
    vec[1:Nb*Nb] = reshape(ud,(Nb*Nb,1))
    vec[Nb*Nb+1:Ntot] = reshape(us,(Nb*Nb,1))
    vec
end

"""
    Reverse operation of vectorize_couple
"""
function unvectorize_couple_ROHF_manifold(vec, N_bds)
    Nb = N_bds[1]; Ntot=2*Nb^2
    ud=reshape(vec[1:Nb*Nb],(Nb,Nb))
    us=reshape(vec[Nb*Nb+1:Ntot],(Nb,Nb))
    ud,us
end

"""
    Express as (Qd,Qs) a tangent vector in XYZ_vec convention
"""
function XYZ_to_TM(XYZ,ΦT,N_bds)
    Nb,Nd,Ns = N_bds; No = Nd+Ns;
    
    X,Y,Z = vec_to_mat(XYZ,N_bds)
    ηd = zeros(Nb,Nb); ηs = zero(ηd);
    # d
    ηd[1:Nd,Nd+1:No]= X; ηd[1:Nd,No+1:Nb] = Y;
    ηd = ηd + ηd'
    ηd = ΦT*ηd*ΦT'
    # s
    ηs[1:Nd,Nd+1:No]= -X; ηs[Nd+1:No,No+1:Nb] = Z;
    ηs = ηs + ηs'
    ηs = ΦT*ηs*ΦT'

    ηd,ηs
end

"""
    Reverse to XYZ_to_TM
"""
function TM_to_XYZ(Q, ΦT,N_bds)
    Nb,Nd,Ns = N_bds; No = Nd+Ns;
    raw_Qd, raw_Qs = ΦT'*Q[1]*ΦT, ΦT'*Q[2]*ΦT

    X = raw_Qd[1:Nd,Nd+1:No]
    Y = raw_Qd[1:Nd, No+1:Nb]
    Z = raw_Qs[Nd+1:No, No+1:Nb]

    mat_to_vec(X,Y,Z,N_bds)
end

function transport_dir(raw_dir, ΦT_next)
    new_dir_d = ΦT_next*raw_dir[1]*ΦT_next'
    new_dir_s = ΦT_next*raw_dir[2]*ΦT_next'
    new_dir_d, new_dir_s
end

###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                     MO formalism                     !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###

# """
#     Retraction in MO formalism, given in part II by eq (39)
# """
# function retraction_MOs(p, Φ, N_bds)
#     Nb, Nd, Ns = N_bds; No = Nd+Ns
#     Ψd, Ψs = p; Φd, Φs = split_MOs(Φ, N_bds);
    
#     # d <-> s rotations
#     X = -Φd'Ψs
#     W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';
    
#     # occupied <-> virtual
#     Ψd_tilde = Ψd .- Φd*Φd'*Ψd .- Φs*Φs'*Ψd;
#     Ψs_tilde = Ψs .- Φd*Φd'*Ψs .- Φs*Φs'*Ψs;
#     V1,D,V2 = svd(hcat(Ψd_tilde, Ψs_tilde))
#     Σ = diagm(D)
    
#     return (Φ*V2*cos(Σ) + V1*sin(Σ))*V2' * exp(W)
# end

# function d_zeta(ΦT, p, N_bds)
#     Φd,Φs = split_MOs(ΦT,N_bds); Ψd,Ψs = p
#     Φd*Ψd' + Ψd*Φd', Φs*Ψs' + Ψs*Φs'
# end


###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                        Tests                         !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###

# """
#     Test if given projectors do belong to the ROHF manifold.
# """
# function test_projs(PdT,PsT, N_bds)
#     Nb,Nd,Ns = N_bds
    
#     test = zero(Float64)
#     test += norm(PdT*PdT - PdT) # is PdT a projector ?
#     test += norm(PsT*PsT - PsT) # same for PsT.
#     test += norm(PdT*PsT)       # Are they orthogonal ?
#     test += tr(PdT) - Nd        # Test traces.
#     test -= tr(PsT) - Ns

#     test
# end

# """ 
#     Generate discretisation in α to plot E_ROHF(α).
# """
# function discretize_α(nb_points; grid_type = "Log",
#                         α_min = 0.001, α_max = 1.0)
#     @assert grid_type ∈ ("Log", "Int")
#     tab_α = Float64[]
#     # Test on grid
#     (grid_type=="Log") && (tab_α = 
#         [α_min*exp((i/nb_points)*log(α_max/α_min)) for i in 0:nb_points])
#     (grid_type=="Int") && (tab_α =
#         [α_min+ i*(α_max-α_min)/nb_points for i=0:nb_points])
#     # return tab
#     tab_α
# end

############ VIEUX
# """
#     Project the couple (A,B) on the tangent space at (PdT,PsT)
# """
# function proj_tangent_space_DM(PdT,PsT,A,B)
#     PvT = Symmetric(I - PdT - PsT)
#     ΠdT = sym(PdT*(A-B)*PsT) + 2*sym(PdT*A*PvT)
#     ΠsT = sym(PdT*(B-A)*PsT) + 2*sym(PsT*B*PvT)
#     ΠdT,ΠsT
# end


# """
#     Takes MOs ΦT in orthonormal basis and gives the projected gradient 
#     on corresponding couple of projectors (PdT,PsT) in the ROHF manifold
# """
# function proj_grad_E(ΦT, Sm12, A, H, N_bds)
#     PdT,PsT = compute_densities(ΦT, N_bds)
#     FdT,FsT = compute_Fock_operators(ΦT, Sm12, A, H, N_bds)
#     proj_∇E = proj_tangent_space_DM(PdT, PsT, 2*FdT, 2*FsT)
#     proj_∇E
# end

# """
#     One retraction one the RHF Grassmanian manifold given in
#     Appendix B
# """
# function RHF_retraction(A)
#     No = round(Int,tr(A))    
#     Io = diagm(ones(No))
#     U = eigen(-Symmetric(A)).vectors
#     R_A = Symmetric(U[:,1:No]*Io*U[:,1:No]')
# end


# """
#     ROHF retraction applied to the couple (A,B) given in appendix B
# """
# function ROHF_retraction(A,B,N_bds)
#     Nb,Nd,Ns = N_bds
#     @assert size(A,1) == Nb & size(B,1) == Nb

#     # Retract
#     R_sum = RHF_retraction(A+B)
#     R_A = RHF_retraction(R_sum*A*R_sum)
#     R_B = Symmetric(R_sum - R_A)

#     R_A, R_B
# end

# function purify_MOs(Σ, Φ)
#     N_bds, A, S, H, atom_info = read_system(Σ)
#     Nb,Nd,Ns = N_bds; S12 = sqrt(Symmetric(S)); Sm12=inv(S12);
#     ΦT = S12*Φ
    
#     PdT, PsT = compute_densities(ΦT, N_bds)
#     PdT, PsT = ROHF_retraction(PdT, PsT, N_bds)
    
#     Cd = eigen(-Symmetric(PdT)).vectors[:,1:Nd]
#     Cs = eigen(-Symmetric(PsT)).vectors[:,1:Ns]
#     Sm12*hcat(Cd,Cs)
# end
