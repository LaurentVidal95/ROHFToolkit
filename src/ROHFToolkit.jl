module ROHFToolkit

using LinearAlgebra
using LinearMaps
using Printf
using OptimKit
# preconditioning system. Replace maybe by OptimKit routines
using IterativeSolvers
using PyCall
# Import pyscf globaly
const pyscf = PyNULL()
function __init__()
    copy!(pyscf, pyimport("pyscf"))
end

export ChemicalSystem
export ROHFManifold
export ROHFState
export ROHFTangentVector
export reset_system!
export rohf_energy
export rohf_energy!
export orthonormalize_state!
export deorthonormalize_state!
export compute_ground_state
include("common/ChemicalSystem.jl")
include("common/ROHFState.jl")
include("common/energy.jl")
include("common/compute_ground_state.jl")
include("common/DM_manifold_tools.jl")
include("common/MO_manifold_tools.jl")

#### Direct Minimization solvers
include("direct_minimization/preconditioning.jl")
include("direct_minimization/OptimKit_solve.jl")

#### Self consistent field
export scf
export hybrid_scf
export DIIS
export ODA
# export scf_anderson_solver
include("self_consistent_field/effective_hamiltonians.jl")
include("self_consistent_field/scf_solvers.jl")
include("self_consistent_field/scf.jl")
include("self_consistent_field/acceleration.jl")
include("self_consistent_field/callback_info.jl")
export generate_virtual_MOs_T
export generate_molden
include("misc/toolbox.jl")
include("misc/generate_virtual_mos.jl")
include("misc/molden.jl")

end # Module
