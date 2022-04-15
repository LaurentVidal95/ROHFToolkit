using LineSearches

"""
    Armijo linesearch using the retraction in MO formalism eq. (39) part II.
"""

function line_search(ΦT, p,
                     Sm12, A, H, N_bds; atom_info = [],
                     E = zero(Float64),
                     ∇E = (zeros(N_bds[1],N_bds[2]),zeros(N_bds[1],N_bds[3])),
                     max_step = one(Float64),
                     linesearch_type = BackTracking(order=3),
                     ρ = 0.5, c = 1e-4, max_iter = 50, #old
                     )

    # LineSearches.jl objects
    function Φ_Ls(step)
        ΦT_next = retraction_MOs(step .*p, ΦT, N_bds)
        energy_from_MOs_T(ΦT_next, Sm12, A, H, N_bds, atom_info = atom_info)
    end

    function dΦ_Ls(step)
        ΦT_next = retraction_MOs(step .*p, ΦT, N_bds)
        τ_p = transport_MOs_same_dirs(p, step, ΦT, ΦT_next, N_bds)
        ∇E_next = grad_E_MO_metric(ΦT_next, Sm12, A, H, N_bds)
        scal_M(∇E_next, τ_p)
    end

    function ΦdΦ_Ls(step)
        ΦT_next = retraction_MOs(step .*p, ΦT, N_bds)
        E_next, ∇E_next = compute_energy_and_gradient(ΦT_next, Sm12,
                                                      A, H, N_bds, atom_info=atom_info)
        τ_p = transport_MOs_same_dirs(p, step, ΦT, ΦT_next, N_bds)
        E_next, scal_M(∇E_next, τ_p)
    end
    
    dΦ0_Ls = scal_M(∇E,p); linesearch = linesearch_type;
    
    α, E_next = linesearch(Φ_Ls, dΦ_Ls, ΦdΦ_Ls, max_step, E, dΦ0_Ls)
    ΦT_next = retraction_MOs(α .*p, ΦT, N_bds)
    @assert(test_MOs(ΦT_next, N_bds) < 1e-8)
    
    α, E_next, ΦT_next
end

function manual_line_search(ΦT, p,
                            Sm12, A, H, N_bds; atom_info = [],
                            E = zero(Float64),
                            ∇E = (zeros(N_bds[1],N_bds[2]),zeros(N_bds[1],N_bds[3])),
                            max_step = one(Float64),
                            ρ = 0.5, c = 1e-4, max_iter = 50,
                            )

    iter = zero(Int64); α = max_step;
    
    ΦT_next = retraction_MOs(α .*p, ΦT, N_bds)
    E_next = energy_from_MOs_T(ΦT_next, Sm12, A, H, N_bds, atom_info = atom_info)
    
    # Test Armijo rule
    while (E_next - E > c*α*scal_M(p,∇E) ) & (iter < max_iter)
        iter += 1
        α *= ρ
        ΦT_next = retraction_MOs(α .*p, ΦT, N_bds)
        E_next = energy_from_MOs_T(ΦT_next, Sm12, A, H, N_bds, atom_info = atom_info)
        @assert(test_MOs(ΦT_next, N_bds)<1e-8)
    end
    
    α, E_next, ΦT_next
end
