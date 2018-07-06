"""
Script to generate 3D initial conditions of self-gravitating gas
disks around a point mass.

"""

import numpy as np
import matplotlib.pyplot as plt
import disk_models as dm
import sys



if __name__=="__main__":

    Ncells = 1000000
    Mdisk = 0.05
    
    #DISK MODEL
    d = dm.disk3d(sigma_type="similarity_softened",csnd0=0.05,l=1.0,
                  Rc = 13,gamma=1.0,floor=0.0000001,
                  adiabatic_gamma=1.01,
                  #adiabatic_gamma = 1.4,
                  self_gravity = True,
                  central_particle = False,
                  Mcentral_soft = 0.2,sigma_soft=0.2)
    
    # Reescale the disk surface density to guarantee desired total mass.
    Rin, Rout = 5e-3,40.0
    mdisk0=d.compute_disk_mass(Rin,Rout)
    d.sigma_disk.sigma0 = d.sigma_disk.sigma0*Mdisk/mdisk0
    mdisk0=d.compute_disk_mass(Rin,Rout)
    print "Disk mass: % .3f" % mdisk0
    
    # Make some illustrative plots
    R,Sigma = d.evaluate_sigma(Rin,Rout)
    _,cs = d.evaluate_soundspeed(Rin,Rout)
    _,Q = d.evaluate_toomreQ(Rin,Rout)
    plt.plot(R,Sigma)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
  

    # DISK MESH
    mesh = dm.disk_mesh3d(mesh_type="mc",Ncells=Ncells,fill_background=True,
                          fill_center=True,fill_box=True,BoxSize=120,Rout=50)

    # Create SNAPSHOT
    s = dm.snapshot()
    s.create(d,mesh)


    # Write files
    s.write_snapshot(d,mesh)
    s.write_parameter_file(d,mesh)

    plt.show()
    exit()

