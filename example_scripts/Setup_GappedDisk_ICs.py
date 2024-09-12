"""
Script to generate 3D initial conditions of self-gravitating gas
disks around a point mass.

"""

import numpy as np
import matplotlib.pyplot as plt
import disk_models as dm
import sys



if __name__=="__main__":

    Ncells = 1200000
    Mdisk = 0.05
    h0 = 0.1 # disk aspect ratio at R=1
    l = 1.0 # temperature profile index
    Rc = 13 # disk characteristic radius
    
    theta_incl=45.0 # disk inclination in degrees
    
    #DISK MODEL
    d = dm.disk3d(sigma_type="similarity_hole",csnd0=h0*(1.0/Rc)**(0.5*l),
                  l=l, Rc = Rc,gamma=1.0,floor=0.0,
                  adiabatic_gamma=1.001,
                  #adiabatic_gamma =CircumstellarTempProfileIndex 1.4,
                  self_gravity = False,
                  central_particle = False,
                  Mcentral_soft = 0.3,sigma_soft=0.5)
    
    # Reescale the disk surface density to guarantee desired total mass.
    Rin, Rout = 5e-2,50.0
    mdisk0=d.compute_disk_mass(Rin,Rout)
    d.sigma_disk.sigma0 = d.sigma_disk.sigma0*Mdisk/mdisk0
    mdisk0=d.compute_disk_mass(Rin,Rout)
    print "Disk mass: % .3f" % mdisk0
    
    # Make some illustrative plots
    R,Sigma = d.evaluate_sigma(Rin,Rout)
    _,cs = d.evaluate_soundspeed(Rin,Rout)
    _,Q = d.evaluate_toomreQ(Rin,Rout)
    _,dmass = d.evaluate_enclosed_mass(Rin,Rout)
    plt.plot(R,np.array(dmass)*1.0/mdisk0)
    plt.plot(R,Sigma)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
  

    # DISK MESH
    mesh = dm.disk_mesh3d(mesh_type="mc",Ncells=Ncells,fill_background=True,
                          fill_center=True,fill_box=True,BoxSize=120,Rin=Rin,Rout=Rout)

    # Create SNAPSHOT
    s = dm.snapshot()
    s.create(d,mesh)
    s.incline(theta_incl,90.0,mesh)

    
    # Write files
    s.write_snapshot(d,mesh,relax_density_in_input = True)
    s.write_parameter_file(d,mesh)

