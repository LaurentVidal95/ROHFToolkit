using Pkg; Pkg.activate("PATH_TO_ROHFToolkit")
using ROHFToolkit
using LineSearches # new, I use it to manual restart when the linesearch is bad.

CFOUR_ex = "xcasscf"
data = CFOUR_init(CFOUR_ex)
x = CASSCFState(data.mo_numbers, data.mo_coeffs, data.overlap, data.energy)

# Choose linesearch and solver
# A good practice is BackTracking(order=3) linesearch for SD and BFGS
# and HagerZhang() for CG.
solver = ConjugateGradientManual
linesearch = HagerZhang()

# if LBFGS
# solver=LBFGSManual


# Set verbosity for debugging
CASSCF_verbose=false
preconditioner=CASSCF_preconditioner

# Launch optimization
res = compute_ground_state(x; CASSCF=true, CFOUR_ex, CASSCF_verbose,
                           solver, linesearch,
                           ## preconditioner
                           preconditioner,
                           # preconditioned=false # if no preconditioning
                           ## LBFGS
                           # B₀ = CASSCF_LBFGS_init
                           )
