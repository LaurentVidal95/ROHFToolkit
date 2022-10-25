"""
Wraps all ground state computation routines.
"""
function compute_ground_state(ζ::ROHFState; solver=ConjugateGradient, solver_kwargs...)
    # Direct minimization
    if (solver ∈ (GradientDescent, ConjugateGradient, LBFGS))
        return direct_minimization_OptimKit(ζ; solver, solver_kwargs...)
    # Self consistent field
    else
        return self_consistent_field(ζ; solver, solver_kwargs...)
    end
    error("Solver not handled")
end
