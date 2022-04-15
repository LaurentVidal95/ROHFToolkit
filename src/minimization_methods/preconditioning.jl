using ProgressMeter
using LinearMaps

"""
    Compute one approximation of the hessian as well as the gradient
    in vector convention.
"""
function build_prec_grad_system_MOs(Φd,Φs,Fd,Fs,N_bds;
                                    numerical_safety = 1e-2)
    Nb,Nd,Ns = N_bds
    Nn = Nd*Ns + Nb*Nd + Nb*Ns
    
    # Precomputations
    Pd, Ps = Φd*Φd', Φs*Φs'
    Fd_dd = Φd'Fd*Φd; Fd_ss = Φs'Fd*Φs; Fd_ds = Φd'Fd*Φs
    Fs_dd = Φd'Fs*Φd; Fs_ss = Φs'Fs*Φs; Fs_ds = Φd'Fs*Φs

    # right hand term
    b_ds = 2*(Fd_ds - Fs_ds); b_bd = 4Fd*Φd - 4Φd*Fd_dd - 4Φs*Fd_ds';
    b_bs = 4Fs*Φs - 4Φd*Fs_ds - 4Φs*Fs_ss
    b_vec = mat_to_vec(b_ds, b_bd, b_bs)

    # L
    A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, (shift_ds, shift_bd, shift_bs) =
        compute_L_blocs_and_shift(Φd, Φs, Pd, Ps, Fd, Fs,
                                  Fd_dd, Fs_dd, Fd_ss, Fs_ss, N_bds)
    shift = max(shift_ds, shift_bd, shift_bs) + numerical_safety #To close to zero hampers cv

    f = XYZ -> L_for_linear_map(XYZ, Φd, Φs, Pd, Ps, Fd, Fs, Fd_dd, Fd_ss, Fs_dd, Fs_ss,
                                A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, shift, N_bds)
    L = LinearMap(f, Nn)

    # L = build_Sylvester_preconditioner(Φd, Φs, Pd, Ps, Fd, Fs, Fd_dd, Fd_ss, Fs_dd, Fs_ss,
    #                                    A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, shift, N_bds)

    L, b_vec
end



"""
    DOC À COMPLETER

    Computes the "A" and "B" matrices involved in the preconditioning systems
    of the form XA - BX = C.

    Also computes the smallest eigenvalue of XA - BX which is numerically found to
    be the sum of the smallest eigenvalue of A and -B.
    Used to ensure that 
"""
function compute_L_blocs_and_shift(Φd, Φs, Pd, Ps, Fd, Fs, Fd_dd, Fs_dd,
                                   Fd_ss, Fs_ss, N_bds)    
    smallest_eigval(M,N) = eigen(Symmetric(M)).values[1] + eigen(Symmetric(-N)).values[1]
    I = diagm(ones(N_bds[1]));
    tab_shift = Float64[]
    
    # L_ds bloc
    A_ds = 2*(Fd_ss- Fs_ss); B_ds = 2*(Fd_dd - Fs_dd);
    append!(tab_shift, smallest_eigval(A_ds, B_ds))

    # L_bd bloc
    A_bd = -4*Fd_dd; B_bd = -4*(I-Pd-Ps)*Fd;
    append!(tab_shift, smallest_eigval(A_bd, B_bd))

    # L_bs bloc
    A_bs = -4*Fs_ss; B_bs = -4*(I-Pd-Ps)*Fs;
    append!(tab_shift, smallest_eigval(A_bs, B_bs))

    for (i,x) in enumerate(tab_shift)
        (x < 0) && (tab_shift[i] = abs(x))
        (x ≥ 0) && (tab_shift[i] = zero(Float64))
    end
    
    A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, tab_shift
end


function L_for_linear_map(XYZ, Φd, Φs, Pd, Ps, Fd, Fs, Fd_dd, Fd_ss, Fs_dd, Fs_ss,
                          A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, shift, N_bds)
    
    X,Yv,Zv = vec_to_mat_MO(XYZ, N_bds);
    
    L_ds = X*A_ds  .- B_ds*X  .+ shift*X
    L_bd = Yv*A_bd .- B_bd*Yv .+ shift*Yv
    L_bs = Zv*A_bs .- B_bs*Zv  .+ shift*Zv

    L_vec = mat_to_vec(L_ds, L_bd, L_bs)
end

function build_Sylvester_preconditioner(Φd, Φs, Pd, Ps, Fd, Fs, Fd_dd, Fd_ss, Fs_dd, Fs_ss,
                          A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, shift, N_bds)
    
    Nb,Nd,Ns = N_bds
    Nn = Nd*Ns + Nb*Nd + Nb*Ns

    L = zeros(Nn,Nn)
    
    # Bloc L_ds
    for i in 1:Nd*Ns
        vec_int = zeros(Nd*Ns,1)
        vec_int[i] = 1
        X = reshape(vec_int,(Nd,Ns))
        L_ds = X*A_ds - B_ds*X + shift * X
        L[:,i] = mat_to_vec(L_ds,zeros(Nb,Nd), zeros(Nb,Ns))
    end

    for i in 1:Nb*Nd
        vec_int = zeros(Nb*Nd,1)
        vec_int[i] = 1
        Yv = reshape(vec_int,(Nb,Nd))
        L_bd = Yv*A_bd .- B_bd*Yv .+ shift*Yv
        L[:,Nd*Ns + i] = mat_to_vec(zeros(Nd,Ns),L_bd, zeros(Nb,Ns))
    end
    
    for i in 1:Nb*Ns
        vec_int = zeros(Nb*Ns,1)
        vec_int[i] = 1
        Zv = reshape(vec_int,(Nb,Ns))
        L_bs = Zv*A_bs .- B_bs*Zv  .+ shift*Zv
        L[:,Nd*Ns + Nb*Nd + i] = mat_to_vec(zeros(Nd,Ns),zeros(Nb,Nd),L_bs)
    end

    L
end




# ######################## TEST


function build_and_store_hessians(Σ::chemical_system, ΦT; output_file = "hessian.dat")
    
    N_bds, A, S, H, atom_info = read_system(Σ)
    S12=sqrt(Symmetric(S)); Sm12=inv(S12);
    
    Nb,Nd,Ns = N_bds
    No = Nd + Ns; Nv = Nb - No
    Nn = Nd*Ns + Nd*Nv + Ns*Nv
   
    progress = Progress(Nn, desc = "Computing hessian ...")
	 
    # MOs and associated densities
    ΦdT = ΦT[:,1:Nd]; PdT=ΦdT*ΦdT';     # Pd
    ΦsT = ΦT[:,Nd+1:No]; PsT=ΦsT*ΦsT';  # Psx
    ΦvT = ΦT[:,No+1:Nb]                 # virtual MOs

    # Fock operators
    FdT,FsT = compute_Fock_operators(PdT, PsT, Sm12, A, H, N_bds)    

    # Hessian
    hessian=zeros(Nn,Nn)

    # Compute all non XYZ-dependant terms    
    dk_Y = zeros(Nd,Nv)
    dk_Z = zeros(Ns,Nv)
    
    for i=1:Nd*Ns
        vec_int=zeros(Nd*Ns,1)      
        vec_int[i]=1.0
        next!(progress)
        dk_X = reshape(vec_int,(Nd,Ns))
        
        Md1=zeros(Nb,Nb)
        Md1[1:Nd,Nd+1:No]=dk_X
        Md1=ΦT*(Md1+Md1')*ΦT'
        
        Ms1=zeros(Nb,Nb)
        Ms1[1:Nd,Nd+1:No]=-dk_X
        Ms1=ΦT*(Ms1+Ms1')*ΦT'

        JMd1, JMs1, KMd1, KMs1 = eval_JT_KT(Md1, Ms1, Sm12, A)

        Ldk_X = 4*(dk_X* ΦsT'*(FdT-FsT)*ΦsT - ΦdT'*(FdT-FsT)*ΦdT *dk_X) + ΦdT'*(4JMd1-2KMd1+2JMs1)*ΦsT
        Ldk_Y = dk_X* ΦsT'*(4FdT-2FsT)*ΦvT + ΦdT'*(8JMd1-4KMd1+4JMs1-2KMs1)*ΦvT
        Ldk_Z = dk_X'*ΦdT'*(2FdT-4FsT)*ΦvT + ΦsT'*(4JMd1-2KMd1+2JMs1-2KMs1)*ΦvT

        vec_out = mat_to_vec(Ldk_X, Ldk_Y, Ldk_Z, N_bds)
        hessian[:,i]=vec_out
    end

    
    dk_X = zeros(Nd,Ns)
    dk_Z = zeros(Ns,Nv)
    
    for i=1:Nd*Nv
        vec_int=zeros(Nd*Nv,1)      
        vec_int[i]=1.0
        
	next!(progress)

        dk_Y = reshape(vec_int,(Nd,Nv))

        Md1=zeros(Nb,Nb)
        Md1[1:Nd,No+1:Nb]=dk_Y
        Md1=ΦT*(Md1+Md1')*ΦT'

        JMd1,KMd1 = eval_JT_KT_one_proj(Md1,Sm12,A)
       
        Ldk_X = dk_Y*ΦvT'*(4FdT-2FsT)*ΦsT + ΦdT'*(4JMd1-2KMd1)*ΦsT
        Ldk_Y =  4*(dk_Y*ΦvT'*FdT*ΦvT-ΦdT'*FdT*ΦdT*dk_Y) + ΦdT'*(8JMd1-4KMd1)*ΦvT
        Ldk_Z = - 2ΦsT'*(FdT+FsT)*ΦdT*dk_Y +  ΦsT'*(4JMd1-2KMd1)*ΦvT
        
        vec_out = mat_to_vec(Ldk_X, Ldk_Y, Ldk_Z, N_bds)
        hessian[:,Nd*Ns+i]=vec_out
    end
    
    dk_X=zeros(Nd,Ns)
    dk_Y=zeros(Nd,Nv)

    for i=1:Ns*Nv
        vec_int=zeros(Ns*Nv,1)      
        vec_int[i]=1.0

	next!(progress)
        
        dk_Z=reshape(vec_int,(Ns,Nv))

        Ms1=zeros(Nb,Nb)
        Ms1[Nd+1:No,No+1:Nb]=dk_Z
        Ms1=ΦT*(Ms1+Ms1')*ΦT'

        JMs1,KMs1 = eval_JT_KT_one_proj(Ms1,Sm12,A)

        Ldk_X = ΦdT'*(2FdT-4FsT)*ΦvT*dk_Z'+  ΦdT'*(2JMs1)*ΦsT
        Ldk_Y =-2ΦdT'*(FdT+FsT)*ΦsT*dk_Z + ΦdT'*(4JMs1-2KMs1)*ΦvT
        Ldk_Z =  4*(dk_Z*ΦvT'*FsT*ΦvT-ΦsT'*FsT*ΦsT*dk_Z) + ΦsT'*(2JMs1-2KMs1)*ΦvT
        
        vec_out = mat_to_vec(Ldk_X, Ldk_Y, Ldk_Z, N_bds)
        hessian[:,Nd*Ns+Nd*Nv+i]=vec_out
    end

    hessian = (hessian + hessian')./2
    # write in file
    
    open("$(output_file)", "w") do f
        for j in 1:size(hessian,1)
            for i in 1:size(hessian,1)
                println(i,j)
                write(f, "$(hessian[i,j]) ")
            end
            write(f,"\n")
        end
    end

    hessian
end


# ######################## VIEUX

# function L_for_linear_map(XYZ, Φd, Φs, Fd, Fs, Fd_dd, Fd_ss, Fd_ds,
#                           Fs_dd, Fs_ss, Fs_ds, N_bds)
    
#     X,Yv,Zv = vec_to_mat_MO(XYZ, N_bds)

#     Pd, Ps = Φd*Φd', Φs*Φs'

#     Nb,Nd,Ns = N_bds #test
    
#     L_ds = 2(X*(Fd_ss-Fs_ss) - (Fd_dd-Fs_dd)*X) + Yv'*(2Fd-Fs)*Φs + (Φd'Fd - 2Φd'Fs)*Zv
#     L_bd = ( (4Fd - 2Fs)*Φs + Φd*(2Fs_ds-4Fd_ds) + Φs*(2Fs_ss-4Fd_ss) )*X' +
#         4(Fd*Yv - (Pd+Ps)*Fd*Yv - Yv*Fd_dd) - 2Zv*(Fd_ds'+Fs_ds') 
#     L_bs  =( (2Fd - 4Fs)*Φd + Φd*(4Fs_dd-2Fd_dd) + Φs*(4Fs_ds'-2Fd_ds') )*X -
#         2Yv*(Fd_ds+Fs_ds) + 4(Fs*Zv - (Pd + Ps)*Fs*Zv - Zv*Fs_ss) 
        
#     L_vec = mat_to_vec(L_ds, L_bd, L_bs)
# end


# function build_approx_hessian_MO(Φd, Φs, Fd, Fs, N_bds)
#     # First computations
#     Nb,Nd,Ns = N_bds
#     Nn = Nd*Ns + Nb*Nd + Nb*Ns

#     # Precomputations
#     Fd_dd = Φd'Fd*Φd; Fd_ss = Φs'Fd*Φs; Fd_ds = Φd'Fd*Φs
#     Fs_dd = Φd'Fs*Φd; Fs_ss = Φs'Fs*Φs; Fs_ds = Φd'Fs*Φs

#     hessian = zeros(Nn,Nn)
    
#     for i in 1:Nd*Ns
#         vec_int = zeros(Nd*Ns,1)
#         vec_int[i] = 1
#         X = reshape(vec_int,(Nd,Ns))

#         H_ds = 2*(X*(Fd_ss - Fs_ss) - (Fd_dd - Fs_dd)*X) 
#         H_bd = ( (4*Fd-2*Fs)*Φs + Φd*(2*Fd_ds - 4*Fd_ds) + Φs*(2*Fs_ss - 4*Fd_ss))*X'
#         H_bs = ( (2*Fd-4*Fs)*Φd + Φd*(4*Fs_dd-2*Fd_dd) + Φs*(4*Fs_ds'- 2*Fd_ds') )*X

#         hessian[:,i] = mat_to_vec(H_ds, H_bd, H_bs)
#     end

#     for i in 1:Nb*Nd
#         vec_int = zeros(Nb*Nd,1)
#         vec_int[i] = 1
#         Yv = reshape(vec_int, (Nb,Nd))

#         H_ds = Yv'*(2*Fd-Fs)*Φs
#         H_bd = 4*(Fd*Yv - Φd*Φd'*Fd*Yv - Φs*Φs'*Fd*Yv - Yv*Fd_dd)
#         H_bs = -2*Yv*(Fd_ds + Fs_ds)

#         hessian[:,Nd*Ns + i] = mat_to_vec(H_ds, H_bd, H_bs)
        
#     end
    
#     for i in 1:Nb*Ns
#         vec_int = zeros(Nb*Ns,1)
#         vec_int[i] = 1
#         Zv = reshape(vec_int, (Nb,Ns))

#         H_ds = (Fd*Φd-2*Fs*Φd)'*Zv
#         H_bd = -2*Zv*(Fd_ds'+Fs_ds')
#         H_bs = 4*(Fs*Zv - Φd*Φd'*Fs*Zv - Φs*Φs'*Fs*Zv - Zv*Fs_ss)

#         hessian[:,Nd*Ns + Nb*Nd + i] = mat_to_vec(H_ds, H_bd, H_bs)
#     end

#     hessian
# end
