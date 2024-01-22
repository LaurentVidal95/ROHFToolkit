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
