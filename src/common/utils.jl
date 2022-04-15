# TODO : CLEAAAAAAAAANN

###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                  General operations                  !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###

"""
    Useful tool, compact wayw to write gradient and projectors.
"""
sym(A::AbstractArray) = Symmetric( 1/2 .*(A .+ transpose(A)) )
"""
    Frobenius scalar product on the ROHF manifold M
"""
scal_M(A,B) = tr(A[1]'B[1]) + tr(A[2]'B[2])

function split_MOs(Φ, N_bds)
    Nb,Nd,Ns = N_bds
    Φd = Φ[:,1:Nd]; Φs = Φ[:,Nd+1:Nd+Ns]
    Φd, Φs
end
split_MOs(ζ::ROHFState) = split_MOs(ζ.Φ, ζ.M.mo_numbers)
function split_MOs(Ψ::ROHFTangentVector)
    mo_numers = Ψ.foot.M.mo_numbers
    Φ = Ψ.foot.Φ
    split_MOs(Ψ.vec, mo_numbers), split_MOs(Φ, mo_numbers)
end

function mat_to_vec(X,Y,Z)
    vec = Float64[]
    for mat in (X,Y,Z)
        for col in eachcol(mat)
            vec = vcat(vec,col)
        end
    end
    vec
end

function vec_to_mat_MO(vec, N_bds)
    Nb,Nd,Ns = N_bds
    Nn = Nd*Ns + Nb*Nd + Nb*Ns
    # reshape
    X=reshape(vec[1:Nd*Ns],(Nd,Ns))
    Y=reshape(vec[Nd*Ns+1:Nd*Ns+Nb*Nd],(Nb,Nd))
    Z=reshape(vec[Nd*Ns+Nb*Nd+1:Nn],(Nb,Ns))
    [X,Y,Z]
end;


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
##!!! !! !  !               ROHF manifold utils                    !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###

"""
    For (Ψd,Ψs) in R^{Nb×Nd}×R^{Nb×Ns} and y = (Φd,Φs) in the MO manifold
    the orthogonal projector on the horizontal tangent space at y is defined by
    
    Π_y(Ψd,Ψs) = ( 1/2*Φs[Φs'Ψd - Ψs'Φd] + Φv(Φv'Ψd),  -1/2*Φd[Ψd'Φs - Φd'Ψs] + Φv(Φv'Ψs) )
"""
function proj_horizontal_tangent_space(ΦT, ΨT, N_bds)
    ΦdT, ΦsT = split_MOs(ΦT, N_bds);
    ΨdT, ΨsT = ΨT
    X = 1/2 .* (ΨdT'ΦsT + ΦdT'ΨsT); I = diagm(ones(N_bds[1]));
    -ΦsT*X' + (I - ΦdT*ΦdT')*ΨdT, -ΦdT*X + (I - ΦsT*ΦsT')*ΨsT
end


###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                     MO formalism                     !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###

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
    Retraction in MO formalism, given in part II by eq (39)
"""
function retraction_MOs(p, Φ, N_bds)
    Nb, Nd, Ns = N_bds; No = Nd+Ns
    Ψd, Ψs = p; Φd, Φs = split_MOs(Φ, N_bds);
    
    # d <-> s rotations
    X = -Φd'Ψs
    W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';
    
    # occupied <-> virtual
    Ψd_tilde = Ψd .- Φd*Φd'*Ψd .- Φs*Φs'*Ψd;
    Ψs_tilde = Ψs .- Φd*Φd'*Ψs .- Φs*Φs'*Ψs;
    V1,D,V2 = svd(hcat(Ψd_tilde, Ψs_tilde))
    Σ = diagm(D)
    
    return (Φ*V2*cos(Σ) + V1*sin(Σ))*V2' * exp(W)
end

"""
    transport p along t*p. [A CORRIGER]
"""
function transport_MOs_same_dirs(p, t, ΦT, ΦT_next, N_bds)
    Nb, Nd, Ns = N_bds; No = Nd+Ns
    Ψd,Ψs = p; Φd, Φs = split_MOs(ΦT, N_bds);
    Φd_next, Φs_next = split_MOs(ΦT_next, N_bds);
    
    # d <-> s rotations
    X = -Φd'*Ψs;
    W = zeros(No,No); W[1:Nd,Nd+1:No] = .- X; W[Nd+1:No,1:Nd] = X';

    # occupied <-> virtual
    Ψd_tilde = Ψd .- Φd*Φd'*Ψd .- Φs*Φs'*Ψd;
    Ψs_tilde = Ψs .- Φd*Φd'*Ψs .- Φs*Φs'*Ψs;
    V1,D,V2 = svd(hcat(Ψd_tilde, Ψs_tilde))
    Σ = diagm(D)

    τ_p = (-ΦT*V2*sin(t .*Σ) + V1*cos( t .*Σ))*Σ*V2' * exp(t .* W) + ΦT_next*W
    Ξd,Ξs = split_MOs(τ_p, N_bds)
    Ξd - Φd_next*Φd_next'Ξd, Ξs - Φs_next*Φs_next'Ξs
end

function d_zeta(ΦT, p, N_bds)
    Φd,Φs = split_MOs(ΦT,N_bds); Ψd,Ψs = p
    Φd*Ψd' + Ψd*Φd', Φs*Ψs' + Ψs*Φs'
end


###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###
##!!! !! !  !                        Tests                         !  ! !! !!!##
###!!!!!!! !! !! !  !                                      !  ! !! !! !!!!!!!###

"""
    Test if new MOs are admissible solutions MAYBE REMOVE THAT !
"""
function test_MOs(ΦT::AbstractArray, N_bds)
    Nb,Nd,Ns = N_bds
    PdT, PsT = compute_densities(ΦT, N_bds)
    
    test = norm(PdT*PdT - PdT)
    test += norm(PsT*PsT - PsT)
    test += norm(PdT*PsT)
    test += tr(PdT) - Nd
    test -= tr(PsT) - Ns

    test
end

"""
    Test if given projectors do belong to the ROHF manifold.
"""
function test_projs(PdT,PsT, N_bds)
    Nb,Nd,Ns = N_bds
    
    test = zero(Float64)
    test += norm(PdT*PdT - PdT) # is PdT a projector ?
    test += norm(PsT*PsT - PsT) # same for PsT.
    test += norm(PdT*PsT)       # Are they orthogonal ?
    test += tr(PdT) - Nd        # Test traces.
    test -= tr(PsT) - Ns

    test
end

""" 
    Generate discretisation in α to plot E_ROHF(α).
"""
function discretize_α(nb_points; grid_type = "Log",
                        α_min = 0.001, α_max = 1.0)
    @assert grid_type ∈ ("Log", "Int")
    tab_α = Float64[]
    # Test on grid
    (grid_type=="Log") && (tab_α = 
        [α_min*exp((i/nb_points)*log(α_max/α_min)) for i in 0:nb_points])
    (grid_type=="Int") && (tab_α =
        [α_min+ i*(α_max-α_min)/nb_points for i=0:nb_points])
    # return tab
    tab_α
end

############ VIEUX
"""
    Project the couple (A,B) on the tangent space at (PdT,PsT)
"""
function proj_tangent_space_DM(PdT,PsT,A,B)
    PvT = Symmetric(I - PdT - PsT)
    ΠdT = sym(PdT*(A-B)*PsT) + 2*sym(PdT*A*PvT)
    ΠsT = sym(PdT*(B-A)*PsT) + 2*sym(PsT*B*PvT)
    ΠdT,ΠsT
end


"""
    Takes MOs ΦT in orthonormal basis and gives the projected gradient 
    on corresponding couple of projectors (PdT,PsT) in the ROHF manifold
"""
function proj_grad_E(ΦT, Sm12, A, H, N_bds)
    PdT,PsT = compute_densities(ΦT, N_bds)
    FdT,FsT = compute_Fock_operators(ΦT, Sm12, A, H, N_bds)
    proj_∇E = proj_tangent_space_DM(PdT, PsT, 2*FdT, 2*FsT)
    proj_∇E
end

"""
    One retraction one the RHF Grassmanian manifold given in
    Appendix B
"""
function RHF_retraction(A)
    No = round(Int,tr(A))    
    Io = diagm(ones(No))
    U = eigen(-Symmetric(A)).vectors
    R_A = Symmetric(U[:,1:No]*Io*U[:,1:No]')
end


"""
    ROHF retraction applied to the couple (A,B) given in appendix B
"""
function ROHF_retraction(A,B,N_bds)
    Nb,Nd,Ns = N_bds
    @assert size(A,1) == Nb & size(B,1) == Nb

    # Retract
    R_sum = RHF_retraction(A+B)
    R_A = RHF_retraction(R_sum*A*R_sum)
    R_B = Symmetric(R_sum - R_A)

    R_A, R_B
end

function purify_MOs(Σ, Φ)
    N_bds, A, S, H, atom_info = read_system(Σ)
    Nb,Nd,Ns = N_bds; S12 = sqrt(Symmetric(S)); Sm12=inv(S12);
    ΦT = S12*Φ
    
    PdT, PsT = compute_densities(ΦT, N_bds)
    PdT, PsT = ROHF_retraction(PdT, PsT, N_bds)
    
    Cd = eigen(-Symmetric(PdT)).vectors[:,1:Nd]
    Cs = eigen(-Symmetric(PsT)).vectors[:,1:Ns]
    Sm12*hcat(Cd,Cs)
end
