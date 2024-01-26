using ROHFToolkit

function test_gradient(x::State, t::T) where {T<:Real}
    @assert x.isortho
    dir = TangentVector(x.Φ*ROHFToolkit.rand_mmo_matrix(x.Σ.mo_numbers), x)
    ∇E = AMO_gradient(x.Φ, x)

    # Compute next point
    Φ_next = retract_AMO(x.Φ, t*dir.vec)
    E_next = energy(x.Σ.Sm12*Φ_next, x)

    # Test
    foo = (E_next - x.energy)/t
    bar = tr(∇E.vec'dir.vec)
    foo, bar, norm(foo-bar)
end

function random_dir(x::State; target_norm=1.)
    dir_vec = x.Φ*ROHFToolkit.rand_mmo_matrix(x.Σ.mo_numbers)
    TangentVector(dir_vec .*( target_norm/norm(dir_vec)), x)
end

function test_transport(x::State, α; target_norm=1)
    dir_1 = random_dir(x; target_norm) # vec to be transported
    dir_2 = random_dir(x; target_norm) # direction of transport
    x_next = State(x, ROHFToolkit.retract_AMO(x.Φ, α*dir_2))
    @show norm(dir_1)
    cos_angle = tr(dir_1.vec'dir_2.vec)/(norm(dir_1)*norm(dir_2))
    @show "cos angle: $(cos_angle)"

    # Transport
    τdir_1 = ROHFToolkit.parallel_transport_AMO(dir_1, x, dir_2, α, x_next)
    
    # Test that the transported vector is in the right tangent space
    norm(τdir_1.vec - ROHFToolkit.project_tangent_AMO(x_next.Φ, x_next.Σ.mo_numbers,
                                                      τdir_1.vec))
end
