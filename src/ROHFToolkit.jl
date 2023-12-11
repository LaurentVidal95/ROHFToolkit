module ROHFToolkit

using LinearAlgebra
using OptimKit         # Riemaniann optimization routines
using LinearMaps       # For preconditioning system. Replace maybe by OptimKit routines
using IterativeSolvers # For preconditioning system. Replace maybe by OptimKit routines
# Handling PySCF
using PyCall
const pyscf = PyNULL() # Import pyscf globaly
function __init__()
    copy!(pyscf, pyimport("pyscf"))
end
using Printf           # nice prints

#### Common data structures and routines
export ChemicalSystem
export ROHFManifold
export ROHFState
export ROHFTangentVector
export reset_state!
export orthonormalize_state!
export deorthonormalize_state!
export compute_ground_state
export generate_molden
include("common/ChemicalSystem.jl")
include("common/ROHFState.jl")
include("common/energy.jl")
include("common/compute_ground_state.jl")
include("common/MO_manifold_tools.jl")
include("common/DM_manifold_tools.jl")
include("common/toolbox.jl")

#### Direct Minimization solvers
export GradientDescent, ConjugateGradient, LBFGS # OptimKit functions
include("direct_minimization/preconditioning.jl")
include("direct_minimization/OptimKit_solve.jl")

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

end # Module
