using LineSearches

"""
    Armijo linesearch using the retraction in MO formalism eq. (39) part II.
"""

function rohf_manifold_linesearch(ζ::ROHFState{T}, p::Matrix{T}, Sm12;
                         E = zero(T), ∇E = zero.(split_MOs(ζ)),
                         max_step = one(Float64),
                         linesearch_type = BackTracking(order=3),
                         ) where {T<:Real}
    # All linesearch routines are performed in orthonormal AOs convention
    @assert(ζ.isortho)
    M = ζ.M
    p_test = ROHFTangentVector(p, ζ)
    # LineSearches.jl objects
    function f(step)
        ζ_next = retract(M, ROHFTangentVector(step .* p, ζ))
        flush(stdout)
        rohf_energy!(ζ_next, Sm12)
    end

    function df(step)
        ζ_next = retract(M, ROHFTangentVector(step .* p, ζ))
        # τ_p = project_tangent!(M, p, ζ_next.Φ)
        τ_p = transport_vec_along_himself(p_test, step, ζ_next)
        ∇E_next = grad_E_MO_metric(ζ_next.Φ, Sm12, ζ_next)
        tr(∇E_next'τ_p.vec)
    end

    function fdf(step)   
        ζ_next = retract(M, ROHFTangentVector(step .* p, ζ))
        E_next, ∇E_next = rohf_energy_and_gradient(ζ_next.Φ, Sm12, ζ_next)
        # τ_p = project_tangent!(M, p, ζ_next.Φ)
        τ_p = transport_vec_along_himself(p_test, step, ζ_next)
        E_next, tr(∇E_next'τ_p.vec)
    end

    # Init objects
    df0 = tr(∇E'p); linesearch = linesearch_type;
    α, E_next = linesearch(f, df, fdf, max_step, E, df0)
    # Actualize ζ and energy
    ζ_next = retract(M, ROHFTangentVector(α.*p, ζ))
    ζ_next.energy = E_next
    
    @assert(test_MOs(ζ_next.Φ, ζ_next.M.mo_numbers) < 1e-8)
    
    α, E_next, ζ_next
end
