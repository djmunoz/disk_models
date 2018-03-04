import numpy as np
import matplotlib.pyplot as plt
import disk_models as dm
import sys



if __name__=="__main__":

    
    #DISK MODEL
    
    d = dm.disk3d()

    

    # DISK MESH
    mesh = dm.disk_mesh3d(mesh_type="spherical",mesh_alignment="interleaved",
                          Rin=0.5,Rout=2.5,
                          NR=100,Nphi=160,Nlat=80,latmax = 30.0*np.pi/180,
                          fill_center=False,fill_box=True,BoxSize=160,
                          N_inner_boundary_rings=0,
                          N_outer_boundary_rings=3)

    # Create SNAPSHOT
    s = dm.snapshot()
    s.create(d,mesh,empty=True)
    #s.incline(37,0,mesh)


    plt.plot(s.gas.pos[:,0],s.gas.pos[:,1],'b.',ms=2.0)
    plt.show()
    
    # Write files
    s.write_snapshot(d,mesh,filename='disk_3d.hdf5')
    s.write_parameter_file(d,mesh)

    exit()
    print "Simulation parameters"
    print "alpha=",alpha
    print "h0=",h0
    print "qb=",qb
    print "eb=",eb
    print "Outer density",sigma_out
    rad = np.sqrt((s.gas.pos[:,0]-mesh.BoxSize*0.5)**2 + (s.gas.pos[:,1]-mesh.BoxSize*0.5)**2)
    ind = np.abs(rad - 1.7) == np.abs(rad - 1.7).min()
    radref0= rad[ind].mean()
    radref1 = rad[rad > radref0].min()
    print "Target mass", s.gas.dens[ind].mean() * (radref1 - radref0) * radref0 * 2 * np.pi / mesh.Nphi
    print s.gas.dens[ind].mean()

    exit()
    
    rad = np.sqrt((s.gas.pos[:,0]-mesh.BoxSize*0.5)**2 + (s.gas.pos[:,1]-mesh.BoxSize*0.5)**2)
    phi = np.arctan2((s.gas.pos[:,1]-mesh.BoxSize*0.5),(s.gas.pos[:,0]-mesh.BoxSize*0.5))
    vphi = -s.gas.vel[:,0] * np.sin(phi) + s.gas.vel[:,1] * np.cos(phi)
    vr = +s.gas.vel[:,0] * np.cos(phi) + s.gas.vel[:,1] * np.sin(phi)
    #ind = np.abs(s.gas.pos[:,2]-mesh.BoxSize*0.5) < 0.2
    ind = s.gas.ids < -2
    plt.plot(rad[ind],s.gas.dens[ind],'g.')
    ind = s.gas.ids == -2
    plt.plot(rad[ind],s.gas.dens[ind],'r.')
    ind = s.gas.ids == -1
    plt.plot(rad[ind],s.gas.dens[ind],'b.')
    ind = s.gas.ids >=0
    plt.plot(rad[ind],s.gas.dens[ind],'k.')
    
    #plt.plot(rad[ind],vr[ind] * s.gas.dens[ind] * rad[ind],'b.')
    #plt.plot(rad[ind],vphi[ind],'b.')
    plt.show()

    exit()
    print s.gas.dens
    print "what"
    #plt.plot(s.gas.pos[:,0],s.gas.pos[:,1],'b.')
    #plt.plot(s.gas.pos[:,0],s.gas.pos[:,2],'b.')
    #plt.xlim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    #plt.ylim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    #plt.show()

    ind = s.gas.ids < -2
    plt.plot(s.gas.pos[ind,0],s.gas.pos[ind,1],'.',color='purple')
    ind = s.gas.ids == -2
    plt.plot(s.gas.pos[ind,0],s.gas.pos[ind,1],'.',color='r')
    ind = s.gas.ids == -1
    plt.plot(s.gas.pos[ind,0],s.gas.pos[ind,1],'.',color='b')
    plt.show()

