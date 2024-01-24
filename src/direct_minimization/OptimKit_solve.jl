@doc raw"""
    direct_minimization_OptimKit(ζ::State; maxiter=500, tol=1e-5,
                                    solver=ConjugateGradient, preconditioned=true,
                                    verbose=true, break_symmetry=false, kwargs...)

Wrapper around OptimKit "optimize" function. The arguments are:
    - ζ: initial point of the optimization on the MO manifold
    - maxiter: maximum number of iterations
    - tol: the convergence is asserted when the gradient norm is bellow tol.
    - solver: optimization method which can be any of the OptimKit routines that doesn't involve
    the MO manifold connection.
    - preconditioned: changes the metric of the MO manifold to compute the gradient, to accelerate
    convergence. Default is true.
    - verbose: true or false.
    - break_symmetry: applies random unitary operation on initial MOs to test convergence
    from random initial guess.
    - kwargs: related to the choice of solver. See OptimKit documentation.
Note that the final State is always returned in non orthonormal AOs convention.
"""
function direct_minimization_OptimKit(ζ::State;
                                      maxiter=500,
                                      tol=1e-5,
                                      solver=ConjugateGradient,
                                      preconditioned=true,
                                      verbose=true,
                                      break_symmetry=false,
                                      fg=energy_and_gradient,
                                      kwargs...)
    # non-orthonormal AO -> orthonormal AO convention
    # TODO add fix in case of high conditioning number
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                 "overlap: $(cond(ζ.Σ.overlap_matrix))")
    orthonormalize_state!(ζ)
    (break_symmetry) && (@warn "Broken symmetry"; ζ.Φ = ζ.Φ*rand_unitary_matrix(ζ))

    # Optimization via OptimKit
    ζ0, E0, ∇E0, _ = optimize(fg, ζ, solver(; gradtol=tol, maxiter, verbosity=0, kwargs...);
                              optim_kwargs(;preconditioned, verbose)...)

    # orthonormal AO -> non-orthonormal AO convention
    deorthonormalize_state!(ζ0)
    (norm(∇E0)>tol) && (@warn "Not converged")
    @info "Final energy: $(E0) Ha"

    (;final_state=ζ0, energy=E0, residual=norm(∇E0))
end

@doc raw"""
   optim_kwargs(;preconditioned=true, verbose=true)

Wraps all the tools needed for Riemannian optimization (retraction, projection,
inner product, etc..) in a format readable by OptimKit.
See `src/common/MO_manifold_tools.jl`
"""
function optim_kwargs(;preconditioned=true, verbose=true)
    kwargs = (; retract, inner, transport!, scale!, add!)
    (verbose) && (kwargs=merge(kwargs, (; finalize!)))
    (preconditioned) && (kwargs=merge(kwargs, (;precondition)))
    kwargs
end

function precondition(ζ::State, η)
    ∇E_prec_vec, converged = preconditioned_gradient_AMO(ζ)
    # If the quasi newton system has not been correctly solved, return
    # un-preconditioned gradient
    if !converged
        return η
    end
    ∇E_prec = TangentVector(∇E_prec_vec, ζ)
    # Return standard gradient if not a descent direction.
    # Avoid errors when starting far from the minimum.
    if (tr(∇E_prec'η)/(norm(∇E_prec)*norm(η))) ≤ 1e-2 # set experimentaly
        @warn "No preconditioning"
        return η
    end
    ∇E_prec
end

function retract(ζ::State{T}, η::TangentVector{T}, α) where {T<:Real}
    @assert(η.base.Φ == ζ.Φ) # check that ζ is the base of η
    # Choose between OMO and AMO
    @assert ζ.virtuals # DEBUG
    Rη = State(ζ, retract_AMO(ζ.Φ, α*η))
    τη = transport_colinear_AMO(η, α, Rη)
    Rη, τη
end

inner(ζ::State, η1::TangentVector, η2::TangentVector) = tr(η1'η2)
@inline function scale!(η::TangentVector{T}, α) where {T<:Real}
    TangentVector(α*η, η.base)
end
@inline function add!(η1::TangentVector{T}, η2::TangentVector{T},
                      α::T2) where {T<:Real, T2<:Real}
    TangentVector(η1 + α*η2, η1.base)
end

function transport!(η1::TangentVector{T}, ζ::State{T},
                    η2::TangentVector{T}, α::T, Rη2::State{T}) where {T<:Real}
    @assert ζ.virtuals # DEBUG
    # Test colinearity
    angle(X::Matrix,Y::Matrix) = tr(X'Y) / (norm(X)*norm(Y))
    colinear_dir = norm(abs(angle(η1.vec, η2.vec)) - 1) < 1e-8
    # Simpler transport for colinear vectors
    if colinear_dir
        @assert η1.base.Φ == η2.base.Φ
        return transport_colinear_AMO(η1, α, Rη2)
    end
    # General transport on the flag manifold
    transport_non_colinear_AMO(η1, ζ, η2, α, Rη2)
end

"""
Equivalent of the prompt routine but with OptimKit conventions
"""
function finalize!(ζ, E, ∇E, n_iter)
    if n_iter == 1
        println("Direct energy minimization")
        println("Initial guess: $(ζ.guess)")
        header = ["Iter", "Energy","log10(ΔE)", "log10(||Π∇E||)"]
        println("-"^58)
        println(@sprintf("%-5s  %-16s  %-16s  %-16s", header...))
        println("-"^58)

        info_out = [n_iter-1, ζ.energy, " "^16, " "^16]
        println(@sprintf("%5i %16.12f %16s %16s", info_out...))
        flush(stdout)
    end
    # Print current iter infos
    log_ΔE = log(10, abs(E  - ζ.energy))
    residual = norm(∇E)
    info_out = [n_iter, E, log_ΔE, log10(residual)]
    println(@sprintf("%5i %16.12f %16.12f %16.12f", info_out...))
    flush(stdout)

    # Actualize energy and history
    ζ.energy = E
    ζ.history = vcat(ζ.history, reshape(info_out, 1, 4))

    # Return entry to match OptimKit.jl conventions
    ζ, E, ∇E
end
