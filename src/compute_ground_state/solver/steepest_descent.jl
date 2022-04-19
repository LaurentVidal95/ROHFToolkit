# TODO PROPER DOCUMENTATION !

function steepest_descent_solver(; preconditioned=true)
    function next_dir(info, Sm12)
        grad = preconditioned ? .- preconditioned_gradient(info.ζ, Sm12) : .- info.∇E
        dir = ROHFTangentVector(grad, info.ζ)
        dir, merge(info,(;dir=dir))
    end
    name = preconditioned ? "Preconditioned Steepest Descent" : "Steepest Descent"

    (;next_dir, name, preconditioned)
end
