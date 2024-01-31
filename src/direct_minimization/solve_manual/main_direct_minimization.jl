@doc raw"""
    OLD: direct_minimization(ζ::State;  TODO)

General direct minimization procedure, which decomposes as such:
    1) Choose a direction according to the method provided in the solver arg.
    2) Linesearch along direction
    3) Check convergence
The arguments are
    TODO
"""
function direct_minimization_manual(ζ::State;
                                    maxiter = 500,
                                    maxstep = 2*one(Float64),
                                    tol = 1e-5,
                                    # Choose solver and preconditioning
                                    solver=ConjugateGradientManual,
                                    preconditioned=true,
                                    preconditioning_trigger=10^(-0.5),
                                    # Type of retraction and transport
                                    retraction=:exp,
                                    transport=:exp,
                                    linesearch,
                                    # Prompt
                                    prompt=default_direct_min_prompt(),
                                    solver_kwargs...)

    # Setup solver and preconditioner
    precondition(ζ) = preconditioned_gradient_AMO(ζ; trigger=preconditioning_trigger)
    sol = solver(; solver_kwargs...)

    # Linesearch.jl only handles Float64 step sisze
    (typeof(maxstep)≠Float64) && (maxstep=Float64(maxstep))

    # non-orthonormal AO -> orthonormal AO convention
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                     "overlap: $(cond(ζ.Σ.overlap_matrix))")
    orthonormalize_state!(ζ)

    # Populate info with initial data
    n_iter          = zero(Int64)
    E, ∇E           = energy_and_riemannian_gradient(ζ)
    E_prev, ∇E_prev = E, ∇E
    dir_vec         = sol.preconditioned ? .- precondition(ζ) : - ∇E
    dir             = TangentVector(dir_vec, ζ)
    step            = zero(Float64)
    converged       = false
    residual        = norm(∇E)

    info = (; n_iter, ζ, E, E_prev, ∇E, ∇E_prev, dir, solver=sol,
            step, converged, tol, residual)

    # init LBFGS solver if needed
    if isa(sol, LBFGSManual)
        B = LBFGSInverseHessian(sol.depth, TangentVector[],  TangentVector[], eltype(E)[])
        info = merge(info, (; B))
    end

    # Display header and initial data
    prompt.prompt(info)

    while (!(info.converged) && (n_iter < maxiter))
        n_iter += 1

        # find next point ζ on ROHF manifold
        step, E, ζ = AMO_linesearch(ζ, dir; E, ∇E, maxstep,
                                    linesearch_type=linesearch,
                                    retraction, transport)

        # Update "info" with the new ROHF point and related quantities
        ∇E = AMO_gradient(ζ)
        ∇E_prev = info.∇E; E_prev = info.E
        residual = norm(info.∇E)
        (residual<tol) && (converged=true)

        info = merge(info, (; ζ, E, E_prev, ∇E, ∇E_prev, residual,
                            n_iter, step, converged))
        prompt.prompt(info)

        # Choose next dir according to the solver and update info with new dir
        dir, info = next_dir(sol, info; precondition, transport)
    end
    # Go back to non-orthonormal AO convention
    deorthonormalize_state!(ζ)
    info = merge(info, (;ζ))

    (info.converged) ? (@info "Final energy: $(ζ.energy) Ha") :
        println("----Maximum iteration reached")

    prompt.clean(info)
end
