import numpy as np
import matplotlib.pyplot as plt
import disk_3d_models.disk_structure as d3d
import sys



if __name__=="__main__":

    #DISK MODEL
    d = d3d.disk(sigma_type="similarity_cavity",csnd0=0.12,l=1.0,
                 R_cav=2.5,xi=3.1,Rout=15,adiabatic_gamma=1.00001)

    #DISK MESH
    mesh = d3d.disk_mesh(mesh_type="mc",Ncells=500000,fill_background=True,
                         fill_center=True,fill_box=True,BoxSize=50)
    #mesh.create(d)

    s = d3d.snapshot()
    s.create(d,mesh)
    #s.incline(37,0,mesh)
    print s.params.targetmass
    print s.params.maxvol

    s.write_snapshot(d,mesh)

    print s.pos[:,0].min(),s.pos[:,1].min(),s.pos[:,2].min()
    print s.pos[:,0].max(),s.pos[:,1].max(),s.pos[:,2].max()
    
    #plt.plot(s.pos[:,0],s.pos[:,1],'b.')
    #plt.plot(s.pos[:,0],s.pos[:,2],'b.')
    #plt.xlim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    #plt.ylim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    #plt.show()
    
