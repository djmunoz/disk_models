import numpy as np
import matplotlib.pyplot as plt
import disk_3d_models.disk_structure as d3d
import sys



if __name__=="__main__":

    #DISK MODEL
    d = d3d.disk(sigma_type="similarity_cavity",csnd0=0.2)

    #DISK MESH
    mesh = d3d.disk_mesh(mesh_type="mc",Ncells=10000,fill_background=True,
                         fill_center=True,fill_box=True,BoxSize=50)
    #mesh.create(d)

    s = d3d.snapshot()
    s.create(d,mesh)
    
    print "what"
