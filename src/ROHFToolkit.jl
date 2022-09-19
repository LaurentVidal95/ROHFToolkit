module ROHFToolkit

using LinearAlgebra
using LinearMaps
using IterativeSolvers
using Preconditioners
using Optim

using Printf
using Dates

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
include("ChemicalSystem.jl")
include("ROHFState.jl")
include("energy.jl")
include("compute_ground_state.jl")

#### Direct Minimization
export steepest_descent
export conjugate_gradient
include("direct_minimization/main_direct_minimization.jl")
include("direct_minimization/linesearch.jl")
include("direct_minimization/preconditioning.jl")
include("direct_minimization/direct_min_solvers.jl")

#### Self consistent field
export scf_nlsolve_solver
export scf_diis
include("self_consistent_field/effective_hamiltonians.jl")
include("self_consistent_field/scf_solvers.jl")
include("self_consistent_field/main_scf.jl")

export generate_virtual_MOs_T
export generate_molden
include("misc/toolbox.jl")
include("misc/generate_virtual_mos.jl")
include("misc/pyscf.jl")
include("misc/prompt_info.jl")

end # Module
