## activate package
ROHFPATH = "/home/..."
using Pkg: Pkg.activate(ROHFPATH)
using ROHFToolkit

# Extract initial data
CFOUR_ex="xcasscf"
data = CFOUR_init(CFOUR_ex)

# Construct initial state
Nb, Ni, Na = data.mo_numbers
Φₒ = data.mo_coeffs*Matrix(I, Nb, Ni+Na)
x_init = ROHFToolkit.CASSCFState(data.mo_numbers, Φₒ, data.overlap,
                                 data.energy)

# Compute energy landscape
# orthonormalize_state!(x_init)
# max_step=1e-4
# N_step = 100
# _, ∇E_init = CASSCF_energy_and_gradient(x_init; CFOUR_ex)
# E_landscape, steps = energy_landscape(x_init, ∇E_init;
#                                       max_step,
#                                       N_step)
# p = plot(steps, E_landscape, xlabel="step size", ylabel="energy")
# plot!(p, size=(600,400), linewidth=2, label=:none)
# title!(p, "energy landscape along gradient")
# savefig!(p, "energy_landscape.pdf")

# Launch optimization
solver = GradientDescent
res = compute_ground_state(x_init; solver,
                           CASSCF=true,
                           CFOUR_ex,
                           preconditioned=false,
                           verbose=false)
