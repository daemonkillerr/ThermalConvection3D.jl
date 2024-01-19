using Literate

directory_of_this_file = @__DIR__
Literate.markdown("../scripts/bin_io_script.jl", "md/", execute=true, documenter=false, credit=false)