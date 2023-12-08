using LineSearches

@doc raw"""
        OLD: rohf_manifold_linesearch(ζ::ROHFState{T}, p::Matrix{T};
                                      E, ∇E, linesearch_type,
                                      max_step = one(Float64),
                                      ) where {T<:Real}

Armijo linesearch using the retraction in orthonormal MO formalism eq. (39) part II.
The ``linesearch_type`` can be any of the linesearch algorithms in the LineSearches.jl
library.
"""
function rohf_manifold_linesearch(ζ::ROHFState{T}, p::Matrix{T};
                                  E, ∇E, linesearch_type,
                                  max_step = one(Float64),
                                  ) where {T<:Real}
    # All linesearch routines are performed in orthonormal AOs convention
    @assert(ζ.isortho)
    M = ζ.M
    
    # LineSearches.jl objects
    p_test = ROHFTangentVector(p, ζ)

    function f(step)
        ζ_next = retract(ROHFTangentVector(step .* p, ζ))
        rohf_energy!(ζ_next)
    end

    function df(step)
        ζ_next = retract(ROHFTangentVector(step .* p, ζ))
        τ_p = transport_vec_along_himself(p_test, step, ζ_next)
        ∇E_next = grad_E_MO_metric(ζ_next.Φ, ζ_next)
        tr(∇E_next'τ_p)
    end

    function fdf(step)
        ζ_next = retract(ROHFTangentVector(step .* p, ζ))
        E_next, ∇E_next = rohf_energy_and_gradient(ζ_next.Φ, ζ_next)
        τ_p = transport_vec_along_himself(p_test, step, ζ_next)
        E_next, tr(∇E_next'τ_p)
    end

    # Init objects
    df0 = tr(∇E'p);
    α, E_next = linesearch_type(f, df, fdf, max_step, E, df0)

    # Actualize ζ and energy
    ζ_next = retract(ROHFTangentVector(α.*p, ζ))
    ζ_next.energy = E_next
    @assert (test_MOs(ζ_next.Φ, ζ_next.M.mo_numbers) < 1e-8)

    α, E_next, ζ_next
end
