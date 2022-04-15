using Test
using ROHFToolkit

@testset "ROHFToolkit.jl" begin
    target_energy = -74.778234
    O = chemical_system(1, data_dir_name = "oxygen_data/")
    
    # SD
    @info "Testing steepest descent"
    rohf_steepest_descent(O, save_MOs = true, Σ_name = "SD", MOs_dir = "out",
                          verbosity ="none")
    @test isfile("out/MOs_SD_1.dat")
    @test (abs(O.E_rohf - target_energy) < 1e-5)

    # Preconditionned SD
    @info "Testing preconditionned steepest descent"
    rohf_steepest_descent(O, restart = true, save_MOs = true,
                          Σ_name = "preconditionned_SD", verbosity ="none")
    @test isfile("out/MOs_preconditionned_SD_1.dat")
    @test (abs(O.E_rohf - target_energy) < 1e-5)


    # ODA
    @info "Testing ODA"
    rohf_ODA(O, restart = true, save_MOs = true, Σ_name = "ODA",
             verbosity ="none")
    @test isfile("out/MOs_ODA_1.dat")
    @test (abs(O.E_rohf - target_energy) < 1e-5)


    # Newton
    @info "Testing Newton"
    rohf_newton(O, restart = true, save_MOs = true, Σ_name = "Newton",
                verbosity ="none")
    @test isfile("out/MOs_Newton_1.dat")
    @test (abs(O.E_rohf - target_energy) < 1e-5)

    # SCF_DIIS
    @info "Testing SCF - AD DIIS"
    rohf_SCF_DIIS(O, restart = true, save_MOs = true, Σ_name = "SCF_DIIS",
                  verbosity ="none")
    @test isfile("out/MOs_SCF_DIIS_1.dat")
    @test (abs(O.E_rohf - target_energy) < 1e-5)
    
    rm("out", recursive = true)
end
