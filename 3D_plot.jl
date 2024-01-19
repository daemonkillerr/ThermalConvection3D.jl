using GLMakie

"""
I/O helper method

Load the final state of the temperature property array 'out_T.bin', which was previously stored in Float32 for plotting.
"""
function load_array(Aname, A)
	fname = string(Aname, ".bin")
	fid = open(fname, "r")
	read!(fid, A)
	close(fid)
end

function visualise(i)
	lx, ly, lz = 3, 1, 1

	nx, ny, nz = 96*2 * 3 - 3*2, 96*2 - 3*2, 96*2 - 3*2
	

	T = zeros(Float32, nx, ny, nz)

	# load data
	load_array("out_T"*string(i,pad=3), T)


	xc, yc, zc = LinRange(-lx/2, lx/2, nx), LinRange(0, ly, ny), LinRange(0, 3*lz, nz)
	fig        = Figure(size = (1600, 1000), fontsize = 24)
	ax         = Axis3(fig[1, 1]; aspect = (3, 1, 1), title = "Temperature")
	surf_T     = contour!(ax, xc, yc, zc, T; alpha = 0.05, colormap = :inferno)
	save("T_3D"*string(i,pad=3)*".png", fig)
	return fig
end

for i = 1:300
	visualise(i)
end