module ROHFToolkit

using LinearAlgebra
using LinearMaps
using IterativeSolvers
using Preconditioners
using Optim
using PyCall

using Printf
using Dates

export ChemicalSystem
export ROHFManifold
export ROHFState
export ROHFTangentVector
export reset_system!
export rohf_energy
include("ChemicalSystem.jl")
include("ROHFManifold.jl")
include("energy.jl")

export generate_virtual_MOs_T
include("misc/utils.jl")
include("misc/generate_virtual_mos.jl")
include("misc/pyscf.jl")

#### ALGORITHMS
export minimize_rohf_energy
include("compute_ground_state/info_prompt.jl")
include("compute_ground_state/linesearch.jl")
include("compute_ground_state/direct_minimization.jl")
include("compute_ground_state/rohf_manifold_methods.jl")

export steepest_descent_solver
export conjugate_gradient_solver
include("compute_ground_state/solver/preconditioning.jl")
include("compute_ground_state/solver/steepest_descent.jl")
include("compute_ground_state/solver/conjugate_gradient.jl")

end # Module
