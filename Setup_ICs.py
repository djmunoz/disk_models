import numpy as np
import matplotlib.pyplot as plt
import disk_3d_models.disk_structure as d3d
import sys



if __name__=="__main__":

    #DISK MODEL
    d = d3d.disk(sigma_type="similarity_cavity",csnd0=0.2)

    #DISK MESH
    mesh = d3d.disk_mesh(mesh_type="mc",Ncells=10000,fill_background=True,
                         fill_center=True,fill_box=True,BoxSize=40)
    #mesh.create(d)

    s = d3d.snapshot()
    s.create(d,mesh)
    s.incline(60,0,mesh)

    plt.plot(s.pos[:,0],s.pos[:,1],'b.')
    plt.xlim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    plt.ylim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    plt.show()
    
