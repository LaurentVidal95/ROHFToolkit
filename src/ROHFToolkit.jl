module ROHFToolkit

using LinearAlgebra
using OptimKit         # Riemaniann optimization routines
using LinearMaps       # For preconditioning system. Replace maybe by OptimKit routines
using IterativeSolvers # For preconditioning system. Replace maybe by OptimKit routines
# Handling PySCF
using PyCall
using DelimitedFiles
const pyscf = PyNULL() # Import pyscf globaly
function __init__()
    copy!(pyscf, pyimport("pyscf"))
end
using Printf           # nice prints
using ProgressMeter

#### Common data structures and routines
export ChemicalSystem
export ROHFManifold
export State
export TangentVector
export reset_state!
export orthonormalize_state!
export deorthonormalize_state!
export generate_molden
include("common/ChemicalSystem.jl")
include("common/State.jl")
include("common/ROHF_energy_gradient.jl")
include("common/OMO_manifold_tools.jl")
include("common/AMO_manifold_tools.jl")
include("common/DM_manifold_tools.jl")
include("common/toolbox.jl")

#### Wrapper around all minimization routines
export compute_ground_state
include("compute_ground_state.jl")

#### Direct Minimization solvers
export GradientDescent, ConjugateGradient, LBFGS # OptimKit functions
include("direct_minimization/ROHF_preconditioner.jl")
include("direct_minimization/OptimKit_wrapper.jl")

export GradientDescentManual, ConjugateGradientManual
export LBFGSManual # Bugged
include("direct_minimization/manual_solvers/AMO_linesearch.jl")
include("direct_minimization/manual_solvers/direct_min_solvers.jl")
include("direct_minimization/manual_solvers/main_direct_minimization.jl")
include("direct_minimization/manual_solvers/prompt_info.jl")

#### Self consistent field
export scf
export hybrid_scf
export DIIS
export ODA
include("self_consistent_field/effective_hamiltonians.jl")
include("self_consistent_field/scf_solvers.jl")
include("self_consistent_field/scf.jl")
include("self_consistent_field/acceleration.jl")
include("self_consistent_field/callback_info.jl")

# CASSCF with cfour
export CASSCF_energy_and_gradient
export CFOUR_init
export CASSCFState
export energy_landscape
export CASSCF_preconditioner
export CASSCF_LBFGS_init
include("cfour.jl")

end # Module
