## activate package
ROHFPATH = "/home/..."
using Pkg: Pkg.activate(ROHFPATH)
using ROHFToolkit

# initialize state
CFOUR_ex="xccasscf"
init_data = CFOUR_init(CFOUR_ex)
x_init = ROHFToolkit.CASSCFState(data[1:4]...)
solver = GradientDescent
res = compute_ground_state(x_init; solver, CASSCF=true, CFOUR_ex)
