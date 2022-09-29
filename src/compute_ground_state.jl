"""
Wraps all ground state computation routines.
"""
function compute_ground_state(ζ::ROHFState; solver=ConjugateGradient, kwargs...)
    # Direct minimization
    if (solver ∈ (GradientDescent, ConjugateGradient, LBFGS))
        return direct_minimization_OptimKit(ζ; solver, kwargs...)
    # Self consistent field
    else
        return self_consistent_field(ζ; solver, kwargs...)
    end
    error("Solver not handled")
end
