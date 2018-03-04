import numpy as np
import matplotlib.pyplot as plt
import disk_models as dm
import sys
import disk_data_analysis.circumbinary as dda

'''
Script to general a disk model in 2D with a non-axisymmetric
perturbation to the density distribution


'''


if __name__=="__main__":


    alpha = 0.01
    h0 = 0.05
    
    #DISK MODEL
    Mdot_out = 1.0
    Rin, Rout = 0.4, 50.0
    def nu(R):
        GM = 1.0
        return alpha * h0**2 * np.sqrt(GM) * R**(0.5)

    sigma_type = "similarity_cavity"
    sigma0,gamma,Rc,xi,R_cav = 1.0 , 0.5, 20.0, 3.0, 0.5
    sigma_out = dm.similarity_cavity_sigma(Rout,sigma0,gamma,Rc,xi,R_cav)
    sigma_ref = dm.similarity_cavity_sigma(3.0,sigma0,gamma,Rc,xi,R_cav)

    r_vals = np.logspace(np.log10(Rin),np.log10(Rout),600)
    sigma_vals = dm.similarity_cavity_sigma(r_vals,sigma0,gamma,Rc,xi,R_cav)
    plt.plot(r_vals,sigma_vals)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    d = dm.disk2d(sigma_type= sigma_type,
                  sigma0 = sigma0,gamma=gamma,Rc=Rc,xi=xi,R_cav = R_cav,
                  l=1.0, csndR0 = Rc,
                  sigma_floor = 1e-7,
                  sigma_back = sigma_out,
                  csnd0=0.12,Rout=70,adiabatic_gamma=1.00001)

    def perturbation_function(R,phi):
        return sigma0 * 4e-2 * np.exp(-(R-3.0)**2/0.1) * np.exp(-(phi-0.0)**2/1.0)
    d.add_perturbation(perturbation_function)
    
    # DISK MESH
    mesh = dm.disk_mesh2d(mesh_type="polar",Rin=Rin,Rout=Rout,
                          NR=600,Nphi=400,
                          fill_center=True,fill_box=True,BoxSize=120,
                          N_inner_boundary_rings=0, N_outer_boundary_rings=0,
                          self_gravity = True) 


    # Create SNAPSHOT
    s = dm.snapshot()
    s.create(d,mesh)


    # Write files
    snapshot_file = 'disk_pert_000.hdf5'
    s.write_snapshot(d,mesh,filename = snapshot_file)
    s.write_parameter_file(d,mesh)

    # Read snapshot from disk and check the surface density
    snap = dda.get_snapshot_data('disk_pert_',0,['POS','MASS'],parttype=0)

    # get a sense of the dynamical range in radius in the simulation
    Rmin, Rmax = 0.1, 10.0
    NR, Nphi = 200, 600
    grid = dda.grid_polar(NR = NR, Nphi = Nphi,Rmin=Rmin,Rmax= Rmax,scale='log')
    grid.X, grid.Y = grid.X + snap.header.boxsize * 0.5, grid.Y  +  snap.header.boxsize * 0.5
    
    rho_interp = dda.disk_interpolate_primitive_quantities(snap,[grid.X,grid.Y],quantities=['MASS'])[0]
    
    # And now we can plot the density field of this structured grid
    fig = plt.figure(figsize=(5,4.5))
    fig.subplots_adjust(top=0.97,right=0.95,left=0.1,bottom=0.12)
    ax = fig.add_subplot(111)
    ax.scatter(grid.X,grid.Y,c=rho_interp ,lw=0)
    ax.axis([54,66,54,66])
    ax.set_xlabel(r'$x$',size=18)
    ax.set_ylabel(r'$y$',size=18)
    ax.set_aspect(1.0)
    plt.show()
