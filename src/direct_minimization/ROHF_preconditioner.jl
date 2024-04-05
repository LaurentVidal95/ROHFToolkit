function default_preconditioner(∇E::TangentVector; trigger=10^(-1/2))
    (norm(∇E) > trigger) && (return ∇E)
    AMO_preconditioner(∇E)
end

AMO_preconditioner(η::TangentVector; num_safety=1e-6) =
    AMO_preconditioner(Fock_operators(η.base)..., η; num_safety)
function AMO_preconditioner(Fi, Fa, η::TangentVector; num_safety=1e-6)
    κ = η.kappa
    Nb, Ni, Na = η.base.Σ.mo_numbers
    Ne = Nb - (Ni+Na)

    # build quasi newton system
    H = build_approx_ROHF_hessian(Fi, Fa, η.base; num_safety)
    κ_vec = mat_to_vec(κ[1:Ni, Ni+1:Ni+Na], κ[1:Ni, Ni+Na+1:Nb],
                       κ[Ni+1:Ni+Na, Ni+Na+1:Nb])

    # Return standard gradient if to far from a minimum
    function vec_to_κ(XYZ, Ni, Na, Ne)
        X, Y, Z = reshape_XYZ(XYZ, Ni, Na, Ne)
        [zeros(Ni,Ni) X Y; -X' zeros(Na,Na) Z; -Y' -Z' zeros(Ne,Ne)]
    end

    # Compute quasi newton direction by solving the system with BICGStab l=3.
    # BICGStab requires that L is positive definite which is not the case
    # far from a minimum. In practice a numerical hack allows to compute
    # the lowest eigenvalue of L (without diagonalizing) and we apply a level-shift.
    # Otherwise, replace by Minres if L is still non-positive definite.
    XYZ, history = bicgstabl(H, κ_vec, 3, reltol=1e-14, abstol=1e-14, log=true)
    Pκ = vec_to_κ(XYZ, Ni, Na, Ne)     # Convert back to matrix format

    # Return unpreconditioned grad if norm(κ) is too high
    # or if -prec_grad is not a descent direction
    angle_grad_precgrad = tr(Pκ'κ)/(norm(Pκ)*norm(κ))
    test_1 = history.isconverged
    test_2 = angle_grad_precgrad > 1e-2
    if !(test_1 && test_2)
        message = "No preconditioning:"
        !test_1 && (message *=" precgrad system not converged;")
        !test_2 && (message *=" not a descent direction;")
        @warn message
        return η
    end
    TangentVector(Pκ, η.base)
end

function build_approx_ROHF_hessian(Fi, Fa, ζ::State; num_safety=1e-6)
    Nb, Ni, Na = ζ.Σ.mo_numbers
    Ne = Nb - (Ni + Na)
    length_XYZ = Ni*Na + Ni*Ne + Na*Ne

    # Prepare hessian blocs
    Hx¹, Hx², Hy¹, Hy², Hz¹, Hz², shifts =
        build_hessian_blocs_and_shift(ζ.Φ, Fi, Fa, ζ.Σ.mo_numbers)
    shift = max(shifts..., num_safety)
    H_tmp = XYZ -> hessian_as_linear_map(XYZ, Hx¹, Hx², Hy¹, Hy², Hz¹, Hz², shift)
    return LinearMap(H_tmp, length_XYZ)
end
build_approx_ROHF_hessian(ζ::State; num_safety=1e-6) =
    build_approx_ROHF_hessian(Fock_operators(ζ)..., ζ; num_safety)

@doc raw"""
TODO see old paper PartII p.13
"""
function build_hessian_blocs_and_shift(Φ::Matrix, Fi::Matrix, Fa::Matrix, mo_numbers)
    Φi, Φa, Φe = split_MOs(Φ, mo_numbers; virtuals=true)
    # Build Hx¹, Hx² such that Hess(X,Y,Z)ⁱᵃ = X*Hx¹- Hx²*X
    Hx¹ = 2*Φa'*(Fi-Fa)*Φa
    Hx² = 2*Φi'*(Fi-Fa)*Φi
    # Build Hy¹, Hy² such that Hess(X,Y,Z)ⁱᵉ = Y*Hy¹- Hy²*Y
    Hy¹ = 4*Φe'Fi*Φe
    Hy² = 4*Φi'Fi*Φi
    # Build Hz¹, Hz² such that Hess(X,Y,Z)ᵃᵉ = Z*Hz¹- Hz²*Z
    Hz¹ = 4*Φe'Fa*Φe
    Hz² = 4*Φa'Fa*Φa

    # Evaluate smallest Hessian eigenval
    smallest_eigval(M,N) = eigvals(Symmetric(M))[1] + eigvals(Symmetric(-N))[1]
    shifts = map([(Hx¹,Hx²), (Hy¹,Hy²), (Hz¹,Hz²)]) do (M,N)
        λ = smallest_eigval(M,N)
        if λ < 0
            return abs(λ)
        else
            return zero(λ)
        end
        error("Not positive nor negative number is impossible")
    end
    Hx¹, Hx², Hy¹, Hy², Hz¹, Hz², shifts
end

function hessian_as_linear_map(XYZ, Hx¹, Hx², Hy¹, Hy², Hz¹, Hz², shift)
    Ni = size(Hx²,1); Na = size(Hx¹,1); Ne = size(Hz¹,1)
    # Vector to Nb×Nb matrix
    X, Y, Z = reshape_XYZ(XYZ, Ni, Na, Ne)
    Hx = X*Hx¹-Hx²*X .+ shift*X
    Hy = Y*Hy¹-Hy²*Y .+ shift*Y
    Hz = Z*Hz¹-Hz²*Z .+ shift*Z
    # Matrix to vector
    0.5 * mat_to_vec(Hx, Hy, Hz)
end


function diaghess_preconditioner(ζ::TangentVector)
    x = ζ.base
    mf = pyscf.scf.RHF(x.Σ.mol)
    Nb, Ni, Na = x.Σ.mo_numbers
    Ne = Nb - (Ni+Na)
    mo_occ = vcat([2 for _ in 1:Ni], [1 for _ in 1:Na], [0 for _ in 1:(Nb-(Ni+Na))])
    h_diag = pyscf.soscf.newton_ah.gen_g_hop_rohf(mf, x.Φ, mo_occ)[3]

    P = diagm(map(x->1/x, h_diag))

    # Apply preconditioner
    κ = ζ.kappa
    κ_vec = mat_to_vec(κ[1:Ni, Ni+1:Ni+Na], κ[1:Ni, Ni+Na+1:Nb],
                       κ[Ni+1:Ni+Na, Ni+Na+1:Nb])
    κ_prec_vec = P*κ_vec
    function vec_to_κ(XYZ, Ni, Na, Ne)
        X, Y, Z = reshape_XYZ(XYZ, Ni, Na, Ne)
        [zeros(Ni,Ni) X Y; -X' zeros(Na,Na) Z; -Y' -Z' zeros(Ne,Ne)]
    end

    TangentVector(vec_to_κ(κ_prec_vec, Ni, Na, Ne), x)    
end
