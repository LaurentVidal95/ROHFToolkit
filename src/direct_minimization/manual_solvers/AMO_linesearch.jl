@doc raw"""
        AMO_linesearch(ζ::State{T}, p::Matrix{T};
                              E, ∇E, linesearch_type,
                              max_step = one(Float64),
                              ) where {T<:Real}

Linesearch using the retraction in AMO formalism.
The ``linesearch_type`` can be any of the linesearch algorithms in the LineSearches.jl library.
"""
function AMO_linesearch(ζ::State{T}, p::TangentVector{T};
                               E, ∇E, linesearch_type,
                               maxstep = one(Float64),
                               retraction_type=:exp,
                               transport_type=:exp
                               ) where {T<:Real}
    # All linesearch routines are performed in orthonormal AOs convention
    @assert(ζ.isortho)

    # LineSearches.jl objects
    function f(step)
        ζ_next = retract_AMO(ζ, TangentVector(step .* p.vec, ζ); type=retraction_type)
        energy(ζ_next)
    end

    function df(step)
        ζ_next = retract_AMO(ζ, TangentVector(step .* p.vec, ζ); type=retraction_type)
        τ_p = transport_AMO(p, ζ, p, step, ζ_next; type=transport_type, collinear=true)
        ∇E_next = AMO_gradient(ζ_next)
        tr(∇E_next'τ_p)
    end

    function fdf(step)
        ζ_next = retract_AMO(ζ, TangentVector(step .* p.vec, ζ); type=retraction_type)
        E_next, ∇E_next =  energy_and_riemannian_gradient(ζ_next)
        τ_p = transport_AMO(p, ζ, p, step, ζ_next; type=transport_type, collinear=true)
        E_next, tr(∇E_next'τ_p)
    end

    # Init objects
    df0 = tr(∇E'p.vec);
    α, E_next = linesearch_type(f, df, fdf, maxstep, E, df0)

    # Actualize ζ and energy
    
    ζ_next = retract_AMO(ζ, TangentVector(α .* p.vec, ζ); type=retraction_type)
    ζ_next.energy = E_next
    @assert (test_MOs(ζ_next) < 1e-8)

    # α = (α > max_step) ? max_step : α
    α, E_next, ζ_next
end
