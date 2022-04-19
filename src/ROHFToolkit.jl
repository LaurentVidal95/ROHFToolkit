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
include("minimization_routines/info_prompt.jl")
include("minimization_routines/linesearch.jl")
include("minimization_routines/direct_minimization.jl")
include("minimization_routines/rohf_manifold_methods.jl")

#TEST
include("paths.jl")

end # Module
