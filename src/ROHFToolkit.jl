module ROHFToolkit

using LinearAlgebra
using LinearMaps
using IterativeSolvers
using Preconditioners
using Optim
using PyCall

using Printf
using Dates

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
include("ChemicalSystem.jl")
include("ROHFManifold.jl")
include("energy.jl")

export generate_virtual_MOs_T
include("misc/toolbox.jl")
include("misc/generate_virtual_mos.jl")
include("misc/pyscf.jl")

#### Direct Minimization
export minimize_rohf_energy
export steepest_descent_solver
export conjugate_gradient_solver
include("direct_minimization/main_direct_minimization.jl")
include("direct_minimization/prompt_info.jl")
include("direct_minimization/linesearch.jl")
include("direct_minimization/rohf_manifold_methods.jl")
include("direct_minimization/preconditioning.jl")
include("direct_minimization/solver.jl")

### Self consistent field
include("self_consistent_field/scf_utils.jl")

export generate_molden
include("misc/molden.jl")

end # Module
