using ThermalConvection3D
using Test


push!(LOAD_PATH, "../src")

using ThermalConvection3D

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()

    printstyled("Testing ThermalConvection3D.jl\n"; bold=true, color=:white)

    run(`$exename -O3 --startup-file=no --check-bounds=no $(joinpath(testdir, "test3D_multixpu.jl"))`)

    return 0
end

exit(runtests())

  