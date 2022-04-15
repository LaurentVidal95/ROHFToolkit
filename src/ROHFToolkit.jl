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
export reset_system!
export rohf_energy
include("ChemicalSystem.jl")
include("ROHFManifold.jl")
include("energies.jl")

export generate_virtual_MOs_T
include("common/utils.jl")
include("common/generate_virtual_mos.jl")
include("common/pyscf.jl")

#### ALGORITHMS
export rohf_SD
export rohf_CG
export rohf_preconditioned_SD
export rohf_preconditioned_CG
include("minimization_methods/sd.jl")
include("minimization_methods/cg.jl")
include("minimization_methods/preconditioned_sd.jl")
include("minimization_methods/preconditioned_cg.jl")

#TEST
include("paths.jl")

end # Module
