module ROHFToolkit

using LinearAlgebra
import OptimKit as Opt # Test external library for Riemaniann optimization routines
                       # Eventually should also test Manopt.jl
using LinearMaps       # For preconditioning system.
using IterativeSolvers # For preconditioning system.
using DelimitedFiles   # For the CFOUR dummy interface

# Call to pyscf for AO basis generation and eri.
using PyCall
const pyscf = PyNULL() # Import pyscf globaly
function __init__()
    copy!(pyscf, pyimport("pyscf"))
end
using Printf           # nice prints

# Common data structures and routines
export ChemicalSystem
include("ChemicalSystem.jl")

# ROHF state and routines on AMO manifolds.
export State
export TangentVector
export densities
export orthonormalize_state!
export deorthonormalize_state!
include("AMO_manifold/State.jl")
include("AMO_manifold/TangentVector.jl")
include("AMO_manifold/geometric_tools.jl")
include("AMO_manifold/ROHF_AMO_energy_gradient.jl")
include("AMO_manifold/ROHF_AMO_preconditioner.jl")

export GradientDescent, ConjugateGradient, LBFGS
include("direct_minimization/main_direct_minimization.jl")
include("direct_minimization/direct_min_solvers.jl")
include("direct_minimization/AMO_linesearch.jl")
include("direct_minimization/OptimKit_wrapper.jl")

# Wrapper around all minimization routines
export compute_ground_state
include("compute_ground_state.jl")

# Other
include("common/toolbox.jl")
include("common/prompts.jl")

# Self consistent field for ROHF
export scf
export hybrid_scf
export DIIS
export ODA
include("self_consistent_field/DM_manifold.jl")
include("self_consistent_field/effective_hamiltonians.jl")
include("self_consistent_field/scf_solvers.jl")
include("self_consistent_field/main_scf.jl")
include("self_consistent_field/acceleration.jl")

# Dummy interface with CFOUR for CASSCF.
# Only works with custom CFOUR version.
export CFOUR_init
export CASSCFState
export CASSCF_preconditioner
export CASSCF_fixed_diag_preconditioner
include("cfour.jl")

end # Module
