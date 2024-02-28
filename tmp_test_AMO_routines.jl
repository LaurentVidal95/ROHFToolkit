using ROHFToolkit
import ROHFToolkit as ROHF
using JSON3


function rand_κ_matrix(mo_numbers; target_norm=1.)
    Nb, Ni, Na = mo_numbers
    Ne = Nb - (Ni+Na)
    X = rand(Ni, Na)
    Y = rand(Ni, Ne)
    Z = rand(Na, Ne)
    return [zeros(Ni,Ni) X Y; -X' zeros(Na,Na) Z; -Y' -Z' zeros(Ne,Ne)]
end

function random_dir(x::State; target_norm=1., return_κ=false)
    κ = rand_κ_matrix(x.Σ.mo_numbers; target_norm)
    κ = κ .* (target_norm/norm(κ))
    dir = TangentVector(κ, x)
    (return_κ) && (return dir, κ)
    return dir
end

function test_gradient(x::State, t::T) where {T<:Real}
    # error("Adapt to new TangentVector convention")
    @assert x.isortho
    dir = random_dir(x; target_norm=1)
    ∇E = ROHF.ROHF_gradient(x.Φ, x)

    # Compute next point
    x_next = ROHF.retract(x, dir, t)[1]
    E_next = ROHF.ROHF_energy(x_next)

    # Test
    foo = (E_next - x.energy)/t
    bar = tr(∇E.kappa'dir.kappa)
    foo, bar, norm(foo-bar)
end

function test_transport(x::State, α; target_norm=1, type=:exp)
    X1 = random_dir(x; target_norm) # vec to be transported
    X2 = random_dir(x; target_norm) # direction of transport
    x_next = ROHFToolkit.retract_AMO(x, X2, α)
    @assert ROHFToolkit.is_point(x_next)

    cos_angle = tr(X1.kappa'X2.kappa)/(norm(X1)*norm(X2))
    @info norm(X1)
    @info "cos angle: $(cos_angle)"

    # Transport
    τX1 = ROHFToolkit.transport_AMO(X1, x, X2, α, x_next; type)

    # Test that the transported vector is in the right tangent space
    ROHFToolkit.is_tangent(τX1)
end
