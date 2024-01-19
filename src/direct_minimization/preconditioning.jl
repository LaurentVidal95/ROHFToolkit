@doc raw"""
    preconditioned_gradient_MO_metric(Fd₀::Matrix{T}, Fs₀::Matrix{T},
                                           ζ::State{T}) where {T<:Real}

Gradient for the preconditioned MO metric as described in equation (39) of the
documentation. Note that we allow for different Fock operators than the one associated 
to the current state ζ, so that the routines also applies for the preconditioning of the
hybrid SCF direct minimization problem.
"""
function preconditioned_gradient_MO_metric(Fd₀::Matrix{T}, Fs₀::Matrix{T},
                                           ζ::State{T}) where {T<:Real}
    mo_numbers = ζ.Σ.mo_numbers

    # Construct preconditioned grad system
    Φd, Φs = split_MOs(ζ)
    L, b = build_prec_grad_system_MO(Φd, Φs, Fd₀, Fs₀, mo_numbers)

    # Compute quasi newton direction by solving the system with BICGStab l=3.
    # BICGStab requires that L is positive definite which is not the case
    # far from a minimum. In practice a numerical hack allows to compute
    # the lowest eigenvalue of L (without diagonalizing) and we apply a level-shift.
    # Otherwise, replace by Minres if L is still non-positive definite.
    XYvZv, history = bicgstabl(L, b, 3, reltol=1e-14, abstol=1e-14, log=true)

    # Return standard gradient if not converged
    if !history.isconverged
        @warn "Preconditioning linear solver did not converge. No preconditioning."
        X, Yv, Zv = vec_to_mat(-b, mo_numbers)
        return project_tangent(mo_numbers, ζ.Φ, hcat(Φs*X' + Yv, -Φd*X + Zv))
    end

    X, Yv, Zv = vec_to_mat(XYvZv, mo_numbers)
    tmp_mat = hcat(Φs*X' + Yv, -Φd*X + Zv)
    
    # The preconditioned gradient is still in full ambiant space
    # and needs to be projected on the tangent space at current point.
    prec_grad = project_tangent(mo_numbers, ζ.Φ, tmp_mat)
end
preconditioned_gradient_MO_metric(ζ::State) =
    preconditioned_gradient_MO_metric(Fock_operators(ζ)..., ζ)

@doc raw"""
    build_prec_grad_system_MO(Φd::Matrix{T}, Φs::Matrix{T},
                                       Fd::Matrix{T}, Fs::Matrix{T}, mo_numbers;
                                       safety = 1e-6) where {T<:Real}

Build the quasi newton preconditioning system to compute the preconditioned gradient.
The system is defined as ``L⋅X=b``, in which ``L`` is never assembled but define
through matrix-vector products thanks to ``LinearMaps.jl``
The ``safety`` arg is there to ensure that ``L`` is positive definite.
"""
function build_prec_grad_system_MO(Φd::Matrix{T}, Φs::Matrix{T},
                                   Fd::Matrix{T}, Fs::Matrix{T}, mo_numbers;
                                   safety = 1e-6) where {T<:Real}
    Nb, Nd, Ns = mo_numbers
    Nn = Nd*Ns + Nb*Nd + Nb*Ns

    # Precomputations
    Fd_dd = Φd'Fd*Φd; Fd_ss = Φs'Fd*Φs; Fd_ds = Φd'Fd*Φs
    Fs_dd = Φd'Fs*Φd; Fs_ss = Φs'Fs*Φs; Fs_ds = Φd'Fs*Φs

    # b (right hand term)
    b_ds = 2*(Fd_ds - Fs_ds); b_bd = 4Fd*Φd - 4Φd*Fd_dd - 4Φs*Fd_ds';
    b_bs = 4Fs*Φs - 4Φd*Fs_ds - 4Φs*Fs_ss
    b_vec = mat_to_vec(b_ds, b_bd, b_bs)

    # L⋅X (left hand term)
    A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, (shift_ds, shift_bd, shift_bs) =
      compute_L_MO_blocs_and_shift(Φd, Φs, Fd, Fs, Fd_dd, Fs_dd, Fd_ss, Fs_ss, mo_numbers)
    # Add shift to avoid conditioning issues. This part is from numerical experiment only.
    shift = max(shift_ds, shift_bd, shift_bs) + safety
    # shift = iszero(shift) ? zero(Float64) : shift + safety

    f = XYZ -> L_as_linear_map(XYZ, A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, shift, mo_numbers)
    L = LinearMap(f, Nn)

    L, b_vec
end

@doc raw"""
    compute_L_MO_blocs_and_shift(Φd, Φs, Fd, Fs, Fd_dd, Fs_dd,
                                       Fd_ss, Fs_ss, mo_numbers)

Computes the "A" and "B" matrices involved in the preconditioning system
of the form ``XA - BX = C``.

Also computes the smallest eigenvalue of ``XA - BX`` which is numerically found to
be the sum of the smallest eigenvalue of ``A`` and ``-B``. Used as a shift
to correct the conditioning of the system.
"""
function compute_L_MO_blocs_and_shift(Φd, Φs, Fd, Fs, Fd_dd, Fs_dd,
                                   Fd_ss, Fs_ss, mo_numbers)
    Pd, Ps = densities(hcat(Φd, Φs), mo_numbers)
    smallest_eigval(M,N) = eigen(Symmetric(M)).values[1] + eigen(Symmetric(-N)).values[1]
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

@doc raw"""
    L_as_linear_map(XYZ, A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, shift, mo_numbers)

Defines the approximated hessian `L` though a matrix-vector product.
See the ``LinearMaps.jl`` documentation.
"""
function L_as_linear_map(XYZ, A_ds, B_ds, A_bd, B_bd, A_bs, B_bs, shift, mo_numbers)
    X,Yv,Zv = vec_to_mat(XYZ, mo_numbers)

    L_ds = X*A_ds  .- B_ds*X  .+ shift*X
    L_bd = Yv*A_bd .- B_bd*Yv .+ shift*Yv
    L_bs = Zv*A_bs .- B_bs*Zv  .+ shift*Zv

    L_vec = mat_to_vec(L_ds, L_bd, L_bs)
end
