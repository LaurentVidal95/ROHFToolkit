How to launch a computation. Example for the Oxygen.


###### Launching a single computation

1) Declare your system: O = chemical_system(1, data_dir_name = "your_data_dir")
(Same as src_old..)

2) Now all data are contained in the chemical system structure and are actualized when a computation is launched
with the system O as argument. Infos are:

- O.E_rohf = Rohf energy (zero if no computation has been done)
- O.MOs = corresponding MOs (guess if no computation has been launched)

- O.four_index_tensor, O.overlap_matrix, O.core_hamiltonian, O.N_bds, O.atom_info: unmutable caracteristics of the system.

Now to launch a simple steepest descent call
   $rohf_steepest_descent(O, cv_threshold = 1e-5); (cv_threshold = criteria on the norm of the projected gradient of E)
you obtain:

Minimization of the energy on the ROHF manifold
Parameters: max_iter = 500
Stopping criteria: ||Π∇E|| < 1.0e-5

Guess energy : -72.26810880255213 

-----------------------------------------------------------------
Energy            δE                Π∇E_norm          Step   Iter 
-----------------------------------------------------------------
-73.664415177158   1.396306374606   8.018772919729  0.40000     1  | SD
-74.453859704396   0.789444527237   1.254714464780  0.02500     2  | SD
-74.581115545605   0.127255841209   2.683155646177  0.20000     3  | SD
-74.671014931026   0.089899385421   0.584494669911  0.02500     4  | SD
-74.702752849632   0.031737918606   2.100319623254  0.40000     5  | SD
-74.755485425943   0.052732576311   0.265017908597  0.02500     6  | SD
-74.755990185368   0.000504759424   0.907460961349  0.20000     7  | SD
-74.766084845596   0.010094660229   0.178286241437  0.02500     8  | SD
-74.767037207432   0.000952361836   0.824076389140  0.40000     9  | SD

		   (...)

-74.778234217558   0.000000000003   0.000054548872  0.05000    64  | SD
-74.778234217558   0.000000000000   0.000060483505  0.05000    65  | SD
-74.778234217603   0.000000000045   0.000012578306  0.02500    66  | SD
-74.778234217611   0.000000000008   0.000056296322  0.40000    67  | SD
-74.778234217648   0.000000000038   0.000006925498  0.02500    68  | SD
-----------------------------------------------------------------
CONVERGED
-74.778234217648   0.000000000038   0.000006925498  0.02500    68


Process ended at 2021-03-16T15:31:44.297
Computation time: 101 milliseconds

Now call display_chemical_system(O) to see the new informations on O:

<><><><><><><><><><><><><><><><><><><><><><><><><><><>
                  Chemical system
<><><><><><><><><><><><><><><><><><><><><><><><><><><>
+----------+
  Orbitals
+----------+
Number of basis functions: 9
Closed orbitals: 3
Open orbitals: 2

+----------+
   Atoms
+----------+
TODO


+-------------------+
 Convergence History
+-------------------+
Result of last computation: CONVERGED in 68 iterations. Computation time: 101 milliseconds

ROHF energy: -74.77823421764839 Ha
Residual: 6.925497585158217e-6



Note than any computation will now start from the minimal MOs stored in O.
To restart from scratch simply add: "resart = true" as a keyword argument.

$rohf_steepest_descent(O, restart = true).


########### Launching multiple computations

You can do it by hand with multiple calls
a) $ rohf_preconditionned_steepest_descent(O, restart = true, thresh_E = 1e-3, thresh_Π∇E = 1e-2)
b) $ rohf_newton(O)

Or with the Pipe package wish allow you to pass the output of a function directly as input of another. (Use Pkg.add("Pipe"))

$ @pipe O |> rohf_preconditionned_steepest_descent(_, restart = true, verbosity="head", cv_threshold = 1e-2) |> rohf_newton(_, verbosity = "tail");

Use head and tail as verbosity to avoir multiple header and tail.

Computation preconditionned_SD_out.out started at 2021-03-16T15:37:58.49

Minimization of the energy on the ROHF manifold
Guess energy : -72.26810880255213

-------------------------------------------------------------------------------------
Energy            δE                ||Π∇E||          cos(p,∇E)        Step   Iter
-------------------------------------------------------------------------------------
-74.096967548276  -1.828858745724   4.452463873997  -0.825411528384  0.50000     1  | Preconditionned SD
-74.682752487457  -0.585784939181   2.549069532587  -0.997480377592  0.50000     2  | Preconditionned SD
-74.770752815627  -0.088000328170   1.030967431666  -0.976780837959  1.00000     3  | Preconditionned SD
-74.777524721370  -0.006771905743   0.310114307723  -0.894179969333  1.00000     4  | Preconditionned SD
-74.778150291368  -0.000625569998   0.083647914500  -0.991405885228  1.00000     5  | Preconditionned SD
-74.778222677055  -0.000072385687   0.025775692424  -0.902158842202  1.00000     6  | Preconditionned SD
-74.778232492076  -0.000009815021   0.009287792355  -0.958514746193  1.00000     7  | Preconditionned SD
-------------------------------------------------------------------------------------
Energy            δE                ||Π∇E||          cos(p,∇E)        Step   Iter
-------------------------------------------------------------------------------------
-74.778234217664  -0.000001725588   0.003239284207  -0.942816491696  1.00000     8  | Newton
-74.778234217664  -0.000000000000   0.000000393698  -0.990961961258  1.00000     9  | Newton
-------------------------------------------------------------------------------------
CONVERGED
-74.778234217664  -0.000000000000   0.000000393698  -0.990961961258  1.00000     9


Process ended at 2021-03-16T15:37:58.69
Computation time: 183 milliseconds
