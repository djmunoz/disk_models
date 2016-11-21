import numpy as np
import matplotlib.pyplot as plt
import disk_3d_models as d3d
import sys



if __name__=="__main__":

    Ncells = 500000
    Mdisk = 0.01
    
    #DISK MODEL
    d = d3d.disk(sigma_type="similarity_cavity",csnd0=0.12,l=1.0,
                 R_cav=2.5,xi=3.1,p=1.0,#adiabatic_gamma=1.00001,
                 adiabatic_gamma = 1.4,
                 self_gravity = False,
                 central_particle = False)
    Rin, Rout = 1e-3,15.0

    # Reescale the disk surface density
    mdisk0=d.compute_disk_mass(Rin,Rout)
    d.sigma_disk.sigma0 = d.sigma_disk.sigma0*Mdisk/mdisk0
    mdisk0=d.compute_disk_mass(Rin,Rout)
    print "Disk mass: % .3f" % mdisk0
    
    #DISK MESH
    mesh = d3d.disk_mesh(mesh_type="mc",Ncells=Ncells,fill_background=True,
                         fill_center=True,fill_box=True,BoxSize=50)
    #mesh.create(d)

    s = d3d.snapshot()
    s.create(d,mesh)
    #s.incline(37,0,mesh)
    print s.params.reference_gas_part_mass
    print s.params.max_volume

    s.write_snapshot(d,mesh)

    print s.gas.pos[:,0].min(),s.gas.pos[:,1].min(),s.gas.pos[:,2].min()
    print s.gas.pos[:,0].max(),s.gas.pos[:,1].max(),s.gas.pos[:,2].max()

    s.write_parameter_file(d,mesh)

    rad = np.sqrt((s.gas.pos[:,0]-mesh.BoxSize*0.5)**2+(s.gas.pos[:,1]-mesh.BoxSize*0.5)**2)
    ind = np.abs(s.gas.pos[:,2]-mesh.BoxSize*0.5) < 0.2
    plt.plot(rad[ind],s.gas.dens[ind],'b.')
    plt.show()
    #plt.plot(s.gas.pos[:,0],s.gas.pos[:,1],'b.')
    #plt.plot(s.gas.pos[:,0],s.gas.pos[:,2],'b.')
    #plt.xlim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    #plt.ylim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    #plt.show()
    
