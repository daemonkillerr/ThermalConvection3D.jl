# Randomized Heatmap

Generation of a heatmap using randomized data

````julia
using Plots
ENV["GKSwstype"]="nul"


"""
I/O helper method

Stores the final state of the temperature property array as 'out_T.bin' for later plotting as Float32.
"""
function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end


"""
I/O helper method

Load the final state of the temperature property array 'out_T.bin', which was previously stored in Float32 for plotting.
"""
function load_array(Aname,A)
    fname = string(Aname,".bin")
    fid=open(fname,"r"); read!(fid,A); close(fid)
end


"""
Main plotting routine
"""
function main()

    A = rand(3,3)  # generate a 3x3 array A of random numbers
    B = zeros(3,3) # to hold the read-in results

    name = "random_io"
    save_array(name, A)    # save the random array
    load_array(name, B)    # load the stored array in B

    return B
end
````

````
Main.var"##292".main
````

````julia
# Only run this in an interactive session:
if isinteractive()
    B = main()
    display(heatmap(B))
end
````

