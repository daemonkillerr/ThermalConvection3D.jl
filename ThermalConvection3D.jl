const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end
using ImplicitGlobalGrid, Plots, Printf, Statistics, LinearAlgebra
import MPI
#Global reductions
maximum_g(A) = (max_l  = maximum(A); MPI.Allreduce(max_l,  MPI.MAX, MPI.COMM_WORLD))

@parallel function assign!(A::Data.Array, B::Data.Array)
    @all(A) = @all(B)
    return
end

@parallel function compute_error!(Err_A::Data.Array, A::Data.Array)
    @all(Err_A) = @all(Err_A) - @all(A)
    return
end

@parallel function compute_0!(RogT::Data.Array, Eta::Data.Array, ∇V::Data.Array, T::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, ρ0gα::Data.Number, η0::Data.Number, dη_dT::Data.Number, ΔT::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(RogT) = ρ0gα*@all(T)
    @all(Eta)  = η0*(1.0 - dη_dT*(@all(T) + ΔT/2.0))
    @all(∇V)   = @d_xa(Vx)/dx + @d_ya(Vy)/dy + @d_za(Vz)/dz
    return
end

@parallel function compute_1!(Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, σxy::Data.Array, σxz::Data.Array, σyz::Data.Array, Eta::Data.Array, ∇V::Data.Array, Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dτ_iter::Data.Number, β::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(Pt)   = @all(Pt) - dτ_iter/β*@all(∇V)
    @all(τxx) = 2.0*@inn_yz(Eta)*(@d_xi(Vx)/dx  - 1.0/3.0*@inn_yz(∇V))
    @all(τyy) = 2.0*@inn_xz(Eta)*(@d_yi(Vy)/dy  - 1.0/3.0*@inn_xz(∇V))
    @all(τzz) = 2.0*@inn_xy(Eta)*(@d_zi(Vz)/dz  - 1.0/3.0*@inn_xy(∇V))
    @all(σxy) = 2.0*@av_xyi(Eta)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))
    @all(σxz) = 2.0*@av_xzi(Eta)*(0.5*(@d_zi(Vx)/dz + @d_xi(Vz)/dx))
    @all(σyz) = 2.0*@av_yzi(Eta)*(0.5*(@d_zi(Vy)/dz + @d_yi(Vz)/dy))
    return
end

@parallel function compute_2!(Rx::Data.Array, Ry::Data.Array, Rz::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dVzdτ::Data.Array, Pt::Data.Array, RogT::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, σxy::Data.Array, σxz::Data.Array, σyz::Data.Array, ρ::Data.Number, dampX::Data.Number, dampY::Data.Number, dampZ::Data.Number, dτ_iter::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(Rx)    = 1.0/ρ *(@d_xa(τxx)/dx + @d_ya(σxy)/dy + @d_za(σxz)/dz - @d_xi(Pt)/dx)
    @all(Ry)    = 1.0/ρ *(@d_ya(τyy)/dy + @d_xa(σxy)/dx + @d_za(σxz)/dz - @d_yi(Pt)/dy)
    @all(Rz)    = 1.0/ρ *(@d_za(τzz)/dz + @d_xa(σxz)/dx + @d_ya(σyz)/dy - @d_zi(Pt)/dz + @av_zi(RogT))
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)*dτ_iter
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)*dτ_iter
    @all(dVzdτ) = dampZ*@all(dVzdτ) + @all(Rz)*dτ_iter
    return
end

@parallel function update_V!(Vx::Data.Array, Vy::Data.Array, Vz::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dVzdτ::Data.Array, dτ_iter::Data.Number)
    @inn(Vx) = @inn(Vx) + @all(dVxdτ)*dτ_iter
    @inn(Vy) = @inn(Vy) + @all(dVydτ)*dτ_iter
    @inn(Vz) = @inn(Vz) + @all(dVzdτ)*dτ_iter
    return
end

@parallel function compute_qT!(qTx::Data.Array, qTy::Data.Array, qTz::Data.Array, T::Data.Array, DcT::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(qTx) = -DcT*@d_xi(T)/dx
    @all(qTy) = -DcT*@d_yi(T)/dy
    @all(qTz) = -DcT*@d_zi(T)/dz
    return
end

@parallel_indices (ix,iy) function advect_T!(dT_dt::Data.Array, qTx::Data.Array, qTy::Data.Array, T::Data.Array, Vx::Data.Array, Vy::Data.Array, dx::Data.Number, dy::Data.Number)
    if (ix<=size(dT_dt, 1) && iy<=size(dT_dt, 2)) dT_dt[ix,iy] = -((qTx[ix+1,iy]-qTx[ix,iy])/dx + (qTy[ix,iy+1]-qTy[ix,iy])/dy) -
                                                                  (Vx[ix+1,iy+1]>0)*Vx[ix+1,iy+1]*(T[ix+1,iy+1]-T[ix  ,iy+1])/dx -
                                                                  (Vx[ix+2,iy+1]<0)*Vx[ix+2,iy+1]*(T[ix+2,iy+1]-T[ix+1,iy+1])/dx -
                                                                  (Vy[ix+1,iy+1]>0)*Vy[ix+1,iy+1]*(T[ix+1,iy+1]-T[ix+1,iy  ])/dy -
                                                                  (Vy[ix+1,iy+2]<0)*Vy[ix+1,iy+2]*(T[ix+1,iy+2]-T[ix+1,iy+1])/dy   end
    return
end

@parallel function update_T!(T::Data.Array, T_old::Data.Array, dT_dt::Data.Array, dt::Data.Number)
    @inn(T) = @inn(T_old) + @all(dT_dt)*dt
    return
end

@parallel_indices (ix,iy,iz) function no_fluxY_T!(T::Data.Array)
    if (ix==size(T, 1) && iy<=size(T ,2) && iz<=size(T ,3)) T[ix,iy,iz] = T[ix-1,iy,iz] end
    if (ix==1          && iy<=size(T ,2) && iz<=size(T ,3)) T[ix,iy,iz] = T[ix+1,iy,iz] end
    return
end
@parallel_indices (ix,iy,iz) function no_fluxZ_T!(T::Data.Array)
    if (ix==size(T, 1) && iy<=size(T ,2) && iz<=size(T ,3)) T[ix,iy,iz] = T[ix-1,iy,iz] end
    if (ix==1          && iy<=size(T ,2) && iz<=size(T ,3)) T[ix,iy,iz] = T[ix+1,iy,iz] end
    return
end

@parallel_indices (iy,iz) function bc_x!(A::Data.Array)
    A[  1, iy,  iz] = A[    2,   iy,   iz]
    A[end, iy,  iz] = A[end-1,   iy,   iz]
    return
end

@parallel_indices (ix,iz) function bc_y!(A::Data.Array)
    A[ ix,  1,  iz] = A[   ix,    2,   iz]
    A[ ix,end,  iz] = A[   ix,end-1,   iz]
    return
end

@parallel_indices (ix,iy) function bc_z!(A::Data.Array)
    A[ ix,  iy,  1] = A[   ix,   iy,    2]
    A[ ix,  iy,end] = A[   ix,   iy,end-1]
    return
end

##################################################
@views function ThermalConvection2D()
    # Physics - dimentionally independent scales
    ly        = 1.0                # domain extend, m
    η0        = 1.0                # viscosity, Pa*s
    DcT       = 1.0                # heat diffusivity, m^2/s
    ΔT        = 1.0                # initial temperature perturbation K
    # Physics - nondim numbers
    Ra        = 1e7                # Raleigh number = ρ0*g*α*ΔT*ly^3/η0/DcT
    Pra       = 1e3                # Prandtl number = η0/ρ0/DcT
    ar        = 3                  # aspect ratio
    # Physics - dimentionally dependent parameters
    lx        = ar*ly              # domain extend, m
    lz        = lx
    w         = 1e-2*ly            # initial perturbation standard deviation, m
    ρ0gα      = Ra*η0*DcT/ΔT/ly^3  # thermal expansion
    dη_dT     = 1e-10/ΔT           # viscosity's temperature dependence
    # Numerics
    nx, ny, nz = 96*ar-1, 96-1, 96*ar-1      # numerical grid resolutions; should be a mulitple of 32-1 for optimal GPU perf
    iterMax   = 5*10^4             # maximal number of pseudo-transient iterations
    nt        = 3000               # total number of timesteps
    nout      = 10                 # frequency of plotting
    nerr      = 100                # frequency of error checking
    ε         = 1e-4               # nonlinear absolute tolerence
    dmp       = 2                  # damping paramter
    st        = 5                  # quiver plotting spatial step
    # Derived numerics
    me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz) # MPI initialisation
    select_device()                                               # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy, dz = lx/(nx_g()-1), ly/(ny_g()-1), lz/(nz_g()-1)         # cell size
    ρ         = 1.0/Pra*η0/DcT                # density
    dt_diff   = 1.0/4.1*min(dx,dy)^2/DcT      # diffusive CFL timestep limiter
    dτ_iter   = 1.0/6.1*min(dx,dy)/sqrt(η0/ρ) # iterative CFL pseudo-timestep limiter
    β         = 6.1*dτ_iter^2/min(dx,dy)^2/ρ  # numerical bulk compressibility
    dampX     = 1.0-dmp/nx                    # damping term for the x-momentum equation
    dampY     = 1.0-dmp/ny                    # damping term for the y-momentum equation
    dampZ     = 1.0-dmp/nz                    # damping term for the y-momentum equation
    # Array allocations
    T         = @zeros(nx , ny, nz  )
    T        .= Data.Array([ΔT*exp(-((x_g(ix,dx,T)-0.5*lx)/w)^2 -((y_g(iy,dy,T)-0.5*ly)/w)^2 -((z_g(iz,dz,T)-0.5*lz)/w)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)])
    T[:,1  ] .=  ΔT/2.0
    T[:,end] .= -ΔT/2.0
    T_old     = @zeros(nx  ,ny  ,nz)
    Pt        = @zeros(nx  ,ny  ,nz)
    ∇V        = @zeros(nx  ,ny  ,nz)
    Vx        = @zeros(nx+1,ny  ,nz)
    Vy        = @zeros(nx  ,ny+1,nz)
    Vz        = @zeros(nx  ,ny  ,nz+1)
    RogT      = @zeros(nx  ,ny  ,nz)
    Eta       = @zeros(nx  ,ny  ,nz)
    τxx       = @zeros(nx  ,ny-2,nz-2)
    τyy       = @zeros(nx-2,ny  ,nz-2)
    τzz       = @zeros(nx-2,ny-2,nz)
    σxy       = @zeros(nx-1,ny-1,nz-2)
    σxz       = @zeros(nx-1,ny-2,nz-1)
    σyz       = @zeros(nx-2,ny-1,nz-1)
    Rx        = @zeros(nx-1,ny-2,nz-2)
    Ry        = @zeros(nx-2,ny-1,nz-2)
    Rz        = @zeros(nx-2,ny-2,nz-1)
    dVxdτ     = @zeros(nx-1,ny-2,nz-2)
    dVydτ     = @zeros(nx-2,ny-1,nz-2)
    dVzdτ     = @zeros(nx-2,ny-2,nz-1)
    dτVx       = @zeros(nx-1,ny-2,nz-2)
    dτVy       = @zeros(nx-2,ny-1,nz-2)
    dτVz       = @zeros(nx-2,ny-2,nz-1)
    qTx       = @zeros(nx-1,ny-2,nz-2)
    qTy       = @zeros(nx-2,ny-1,nz-2)
    qTz       = @zeros(nx-2,ny-2,nz-1)
    dT_dt     = @zeros(nx-2,ny-2,nz-2)
    ErrP      = @zeros(nx  ,ny  ,nz)
    ErrV      = @zeros(nx  ,ny+1)
    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")
    X, Y, Z   = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2
    Xc, Yc, Zc = [x for x=X, y=Y, z=Z], [y for x=X, y=Y, z=Z], [z for x=X, y=Y, z=Z]
    Xp, Yp, Zp = Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end], Zc[1:st:end,1:st:end]
    # Time loop
    err_evo1=[]; err_evo2=[]
    for it = 1:nt
        @parallel assign!(T_old, T)
        errV, errP = 2*ε, 2*ε; iter=1; niter=0
        while (errV > ε || errP > ε) && iter <= iterMax
            @parallel assign!(ErrV, Vy)
            @parallel assign!(ErrP, Pt)
            @parallel compute_0!(RogT, Eta, ∇V, T, Vx, Vy, ρ0gα, η0, dη_dT, ΔT, dx, dy)
            @parallel compute_1!(Pt, τxx, τyy, σxy, Eta, ∇V, Vx, Vy, dτ_iter, β, dx, dy)
            @parallel compute_2!(Rx, Ry, dVxdτ, dVydτ, Pt, RogT, τxx, τyy, σxy, ρ, dampX, dampY, dτ_iter, dx, dy)
            @parallel update_V!(Vx, Vy, dVxdτ, dVydτ, dτ_iter)
            @parallel (1:size(Vx,1)) bc_y!(Vx)
            @parallel (1:size(Vy,2)) bc_x!(Vy)
            @parallel compute_error!(ErrV, Vy)
            @parallel compute_error!(ErrP, Pt)
            if mod(iter,nerr) == 0
                errV = maximum(abs.(Array(ErrV)))/(1e-12 + maximum(abs.(Array(Vy))))
                errP = maximum(abs.(Array(ErrP)))/(1e-12 + maximum(abs.(Array(Pt))))
                push!(err_evo1, max(errV, errP)); push!(err_evo2,iter)
                # @printf("Total steps = %d, errV=%1.3e, errP=%1.3e \n", iter, errV, errP)
            end
            iter+=1; niter+=1
        end
        # Thermal solver
        @parallel compute_qT!(qTx, qTy, T, DcT, dx, dy)
        @parallel advect_T!(dT_dt, qTx, qTy, T, Vx, Vy, dx, dy)
        dt_adv = min(dx/maximum(abs.(Array(Vx))), dy/maximum(abs.(Array(Vy))))/2.1
        dt     = min(dt_diff, dt_adv)
        @parallel update_T!(T, T_old, dT_dt, dt)
        @parallel no_fluxY_T!(T)
        @printf("it = %d (iter = %d), errV=%1.3e, errP=%1.3e \n", it, niter, errV, errP)
        # Visualization
        if mod(it,nout)==0
            heatmap(X, Y, Array(T)', aspect_ratio=1, xlims=(X[1], X[end]), ylims=(Y[1], Y[end]), c=:inferno, clims=(-0.1,0.1), title="T° (it = $it of $nt)")
            Vxp = 0.5*(Vx[1:st:end-1,1:st:end  ]+Vx[2:st:end,1:st:end])
            Vyp = 0.5*(Vy[1:st:end  ,1:st:end-1]+Vy[1:st:end,2:st:end])
            Vscale = 1/maximum(sqrt.(Vxp.^2 + Vyp.^2)) * dx*(st-1)
            quiver!(Xp[:], Yp[:], quiver=(Vxp[:]*Vscale, Vyp[:]*Vscale), lw=0.1, c=:blue); frame(anim)
            # display( quiver!(Xp[:], Yp[:], quiver=(Vxp[:]*Vscale, Vyp[:]*Vscale), lw=0.1, c=:blue) )
        end
    end
    gif(anim, "ThermalConvect2D.gif", fps = 15)
    return
end

ThermalConvection2D()