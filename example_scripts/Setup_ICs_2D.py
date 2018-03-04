import numpy as np
import matplotlib.pyplot as plt
import disk_models as dm
import sys



if __name__=="__main__":

    # Binary and disk parameters
    qb = float(sys.argv[1])
    eb = float(sys.argv[2])
    alpha = float(sys.argv[3])
    h0 = float(sys.argv[4])
    
    #DISK MODEL
    Mdot_out = 1.0
    Rin, Rout = 1.0, 70.0
    def nu(R):
        GM = 1.0
        return alpha * h0**2 * np.sqrt(GM) * R**(0.5)
    sigma_out = Mdot_out / 3 / np.pi / nu(Rout)
    R_cav, p, xi, l0  = 2.3, 0.5, 3.5, 0.65
    sigma0 = sigma_out * (Rout/R_cav)**p
    def sigma_function(R):
        return sigma0 * (R_cav/R)**p * np.exp(-(R_cav/R)**xi) #* (1 - l0 * np.sqrt(1.0/R))
        #return sigma0 * (R_cav/R)**p 

    sigma_out = sigma_function(Rout)
    mdot_out = 3 * np.pi * nu(Rout) * sigma_out


    print mdot_out
    quadrupole_correction = 0.25 * qb / (1 + qb**2) * (1 + 1.5 * eb**2)
    
    #radii = np.linspace(0.1,20,1500)
    #plt.plot(radii,sigma_function(radii),'r')

    
    '''

    d = dm.disk2d(sigma_type="similarity",
                  l=1.0,p=0.5,
                  csnd0=0.12,Rout=70,adiabatic_gamma=1.00001)
    '''
    
    d = dm.disk2d(sigma_function = sigma_function,
                  constant_accretion = mdot_out,
                  R0 = 1.0, csndR0 = 1.0,
                  sigma0 = sigma0,
                  sigma_floor = 1e-7,
                  sigma_back = sigma_out,
                  csnd0=h0,adiabatic_gamma=1.00001,
                  quadrupole_correction = quadrupole_correction)

    # DISK MESH
    mesh = dm.disk_mesh2d(mesh_type="polar",Rin=1.0,Rout=70.0,
                          NR=600,Nphi=400,
                          fill_center=False,fill_box=True,BoxSize=160,
                          N_inner_boundary_rings=0,
                          N_outer_boundary_rings=3) 


    # Create SNAPSHOT
    s = dm.snapshot()
    s.create(d,mesh)
    #s.incline(37,0,mesh)


    # Write files
    s.write_snapshot(d,mesh,filename='disk_qb%.2f_eb%.2f_alpha%.2f_h%.2f.hdf5' % (qb,eb,alpha,h0))
    s.write_parameter_file(d,mesh)

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

