function preconditioned_gradient_AMO(ζ::State; num_safety=1e-6)
    Fi, Fa = Fock_operators(ζ)
    H, G_vec = build_quasi_newton_system(ζ.Φ, Fi, Fa, ζ.Σ.mo_numbers;
                                         num_safety)

    # Compute quasi newton direction by solving the system with BICGStab l=3.
    # BICGStab requires that L is positive definite which is not the case
    # far from a minimum. In practice a numerical hack allows to compute
    # the lowest eigenvalue of L (without diagonalizing) and we apply a level-shift.
    # Otherwise, replace by Minres if L is still non-positive definite.
    XYZ, history = bicgstabl(H, G_vec, 3, reltol=1e-14, abstol=1e-14, log=true)

    # Convert back to matrix format
    Nb, Ni, Na = ζ.Σ.mo_numbers
    Ne = Nb - (Ni+Na)
    X, Y, Z = reshape_XYZ(XYZ, Ni, Na, Ne)
    κ = [zeros(Ni,Ni) X Y; -X' zeros(Na,Na) Z; -Y' -Z' zeros(Ne,Ne)]

    # Return preconditioned gradient and convergence data
    ζ.Φ*κ, history.isconverged
end

function build_quasi_newton_system(Φ::Matrix, Fi, Fa, mo_numbers;
                                   num_safety=1e-6)
    Nb, Ni, Na = mo_numbers
    Ne = Nb - (Ni + Na)
    length_XYZ = Ni*Na + Ni*Ne + Na*Ne

    # Build the gradient in XYZ convention
    Φi, Φa, Φe = split_MOs(Φ, mo_numbers; virtuals=true)
    Gx = -Φi'*(Fi-Fa)*Φa
    Gy = -2*Φi'Fi*Φe
    Gz = -2*Φa'Fa*Φe
    G_vec = mat_to_vec(Gx, Gy, Gz)

    # Prepare hessian blocs
    Hx¹, Hx², Hy¹, Hy², Hz¹, Hz², shifts =
        build_hessian_blocs_and_shift(Φ, Fi, Fa, mo_numbers)
    shift = max(shifts..., num_safety)
    H_tmp = XYZ -> hessian_as_linear_map(XYZ, Hx¹, Hx², Hy¹, Hy², Hz¹, Hz², shift)
    H = LinearMap(H_tmp, length_XYZ)

    # Return gradient as XYZ vector and hessian as a linear map of XYZ
    H, G_vec
end

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
    mat_to_vec(Hx, Hy, Hz)
end
