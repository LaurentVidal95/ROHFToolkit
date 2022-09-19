"""
Wraps all ground state computation routines.
"""
function compute_ground_state(ζ::ROHFState; solver=steepest_descent(), kwargs...)
    # Direct minimization
    if (true ∈ contains.(Ref(solver.prefix), ("SD", "CG")))
        return direct_minimization(ζ; solver, kwargs...)
    # Self consistent field
    elseif contains(solver.prefix, "SCF")
        return self_consistent_field(ζ; solver, kwargs...)
    end
    error("Solver not handled")
end
