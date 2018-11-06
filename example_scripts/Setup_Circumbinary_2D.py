import numpy as np
import matplotlib.pyplot as plt
import disk_models as dm
import disk_data_analysis.circumbinary as dda
import sys

'''
Script to setup a steadily accreting circumbinary disk
in 2D.

'''


Mdot_out = 1.0
R_out = 140 #70.0
NR = 700 #600
BOX = 300.0

def sigma_function(R,Rcav,sigma0,p,xi,zeta):
    return sigma0 * (1.0/R)**p * np.exp(-(R_cav/R)**xi + (R_cav/R)**zeta) * (1 - l0 * np.sqrt(1.0/R))

def sigma_profile(R,Rcav,p,xi,zeta):
    sigma0 = Mdot_out / 3 / np.pi / alpha / h0**2 
    sigma_try = sigma_function(R,Rcav,sigma0,p,xi,zeta)
    mdot_try = mdot(R,sigma_try)
    mdot0 = mdot_try[(R < 68) & (R > 60)].mean()
    sigma0 /= mdot0
    return sigma_function(R,Rcav,sigma0,p,xi,zeta)

def velr(R,sigma):
    
    Omega = np.sqrt(1.0/R**3 * (1 + 3 * quadrupole_correction/R**2))
    dOmegadR = np.gradient(Omega)/np.gradient(R)
    csnd = h0 / np.sqrt(R)
    visc = alpha * csnd * csnd / Omega
    func1 = visc * sigma * R**3 * dOmegadR
    dfunc1dR = np.gradient(func1)/np.gradient(R)
    func2 = R**2 * Omega
    dfunc2dR = np.gradient(func2)/np.gradient(R)
    velr = dfunc1dR / R / sigma / dfunc2dR

    return velr

def mdot(R,sigma):

    return -2 * np.pi * R * velr(R,sigma) * sigma



if __name__=="__main__":

    # Binary and disk parameters
    qb = float(sys.argv[1])
    eb = float(sys.argv[2])
    alpha = float(sys.argv[3])
    h0 = float(sys.argv[4])
    
    #DISK MODEL
    Mdot_out = 1.0
    Rin, Rout = 2 * (1.0 + eb)/(1+qb), R_out
    if (eb < 0.02):
        R_cav, p, xi,zeta, l0  = 2.5, 0.5, 4.0, 2.0, (0.75 + 0.35 * eb)
    else:
        R_cav, p, xi,zeta,l0  = (1.5 * eb + 2.3), 0.5, 4.0+1.5*eb, 2*(1+2*eb),(0.75 + 0.35 * eb)

    

    quadrupole_correction =  0.25 * qb / (1 + qb**2) * (1 + 1.5 * eb**2)

    # A slight rescaling
    radii = np.logspace(np.log10(Rin),np.log10(Rout),1000)
    sigma_out = sigma_profile(radii,R_cav,p,xi,zeta)[-1]
    sigma0 = Mdot_out / 3 / np.pi / alpha / h0**2
    sigma0 *= sigma_out/sigma_function(Rout,R_cav,sigma0,p,xi,zeta)

    def sigma_model(R):
        Rcav = 2.*R_cav
        return sigma0 * (1.0/R)**p * np.exp(-(Rcav/R)**xi + (Rcav/R)**zeta) * (1 - l0 * np.sqrt(1.0/R))

    d = dm.disk2d(sigma_function = sigma_model,
                  constant_accretion = False,#np.abs(Mdot_out),
                  R0 = 1.0, csndR0 = 1.0,
                  sigma0 = sigma0,
                  sigma_floor = 1e-7,
                  sigma_back = sigma_out/10000,
                  csnd0=h0,adiabatic_gamma=1.00001,
                  alphacoeff = alpha,
                  quadrupole_correction = quadrupole_correction)

    # DISK MESH
    mesh = dm.disk_mesh2d(mesh_type="polar",Rin= Rin,Rout=Rout,
                          mesh_alignment = "interleaved",
                          NR=NR,Nphi=400,
                          #Nphi_inner_bound = 600,
                          fill_center=False,fill_box=True,BoxSize=BOX,
                          N_inner_boundary_rings=2,
                          N_outer_boundary_rings=1) 

    
    
    # Create SNAPSHOT
    s = dm.snapshot()
    s.create(d,mesh)
    #s.incline(37,0,mesh)


    # Write files
    s.write_snapshot(d,mesh,filename='disk_qb%.2f_eb%.2f_alpha%.2f_h%.2f.hdf5' % (qb,eb,alpha,h0),
                     relax_density_in_input = True)
    print s.params.cpu_time_bet_restart_file
    s.params.read('param_example.txt')
    s.params.time_limit_cpu=28000 
    s.params.reference_gas_part_mass=1.1e-4
    s.params.target_gas_mass_factor=1.0    
    s.params.courant_fac=0.25
    s.params.max_size_timestep=0.1
    s.params.min_size_timestep=7.0e-15
    s.params.cell_shaping_speed=0.52
    s.params.cell_max_angle_factor=1.25
    s.params.active_part_frac_for_new_domain_decom=0.85
    s.params.top_node_factor=8
    s.params.multiple_domains=32
    s.params.circumstellar_boundary_density = sigma_model(Rout)
    s.params.inner_radius = Rin
    s.params.outer_radius = Rout
    s.params.binary_eccentricity = eb
    s.params.binary_mass_ratio = qb
    s.write_parameter_file(d,mesh,filename='param_qb%.2f_eb%.2f_alpha%.2f_h%.2f.txt' % (qb,eb,alpha,h0))

    print "Simulation parameters"
    print "alpha=",alpha
    print "h0=",h0
    print "qb=",qb
    print "eb=",eb
    print "InnerRadius=",Rin
    print "Outer density",sigma_model(Rout)
    print "Number of cells:",s.gas.pos.shape[0]
    rad = np.sqrt((s.gas.pos[:,0]-mesh.BoxSize*0.5)**2 + (s.gas.pos[:,1]-mesh.BoxSize*0.5)**2)
    ind = np.abs(rad - 1.7) == np.abs(rad - 1.7).min()
    radref0= rad[ind].mean()
    radref1 = rad[rad > radref0].min()
    print "Target mass", s.gas.dens[ind].mean() * (radref1 - radref0) * radref0 * 2 * np.pi / mesh.Nphi
    


    
    '''
    ind = s.gas.ids < -2
    plt.plot(s.gas.pos[ind,0],s.gas.pos[ind,1],'g.')
    ind = s.gas.ids == -2
    plt.plot(s.gas.pos[ind,0],s.gas.pos[ind,1],'r.')
    ind = s.gas.ids == -1
    plt.plot(s.gas.pos[ind,0],s.gas.pos[ind,1],'b.')
    ind = s.gas.ids >= 0
    plt.plot(s.gas.pos[ind,0],s.gas.pos[ind,1],'k.',ms=1.0)   
    plt.xlim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    plt.ylim(0.5*mesh.BoxSize-mesh.Rout,0.5*mesh.BoxSize+mesh.Rout)
    plt.show()
    exit()
    '''
    

    ind = s.gas.ids < -2
    plt.plot(rad[ind],s.gas.dens[ind],'g.')
    ind = s.gas.ids == -2
    plt.plot(rad[ind],s.gas.dens[ind],'r.')
    ind = s.gas.ids == -1
    plt.plot(rad[ind],s.gas.dens[ind],'b.')
    ind = s.gas.ids >= 0
    plt.plot(rad[ind],s.gas.dens[ind],'k.')
    radii = np.linspace(1,70,500)
    plt.plot(radii,sigma_model(radii),color='red')
    plt.show()

    #exit()

    
    '''
    r, sigma = d.evaluate_sigma(1,70)
    _, vr = d.evaluate_radial_velocity(1,70)
    vphi = d.evaluate_rotation_curve(1,70)[1] * r

    jdotadv = -2 * np.pi * r**2 * vr * vphi * sigma
    jdotvisc = -2 * np.pi * r**(3.5) * alpha * h0**2 * sigma * \
               d.evaluate_radial_gradient(vphi/r,1.0,70)[1]

    vrsteady= -Mdot_out/2/np.pi/r/sigma/r**2/(vphi/r) * l0 + \
              alpha  *h0**2 * r**0.5 / (vphi/r) * d.evaluate_radial_gradient(vphi/r,1.0,70)[1]

    vrmean = 0.5 * (vr + vrsteady)
    plt.plot(r,vr)
    #plt.plot(r,vrsteady)
    plt.ylim(-0.004,0.001)
    #plt.plot(r,np.abs(vr-vrsteady)/np.abs(vrmean))
    #plt.ylim(0,0.05)
    plt.show()
    exit()
    '''
    
    rad = np.sqrt((s.gas.pos[:,0]-mesh.BoxSize*0.5)**2 + (s.gas.pos[:,1]-mesh.BoxSize*0.5)**2)
    phi = np.arctan2((s.gas.pos[:,1]-mesh.BoxSize*0.5),(s.gas.pos[:,0]-mesh.BoxSize*0.5))
    vphi = -s.gas.vel[:,0] * np.sin(phi) + s.gas.vel[:,1] * np.cos(phi)
    vr = +s.gas.vel[:,0] * np.cos(phi) + s.gas.vel[:,1] * np.sin(phi)
    mdot = -vr * s.gas.dens * 2 * np.pi * rad
    
    '''
    ind = s.gas.ids < -2
    plt.plot(rad[ind],vphi[ind] ,'g.')
    ind = s.gas.ids == -2
    plt.plot(rad[ind],vphi[ind],'r.')
    ind = s.gas.ids == -1
    plt.plot(rad[ind],vphi[ind],'b.')
    ind = s.gas.ids >= 0
    plt.plot(rad[ind],vphi[ind],'k.')
    plt.show()
    exit()
    '''


    ind = s.gas.ids < -2
    plt.plot(rad[ind],vr[ind] ,'g.')
    ind = s.gas.ids == -2
    plt.plot(rad[ind],vr[ind],'r.')
    ind = s.gas.ids == -1
    plt.plot(rad[ind],vr[ind],'b.')
    ind = s.gas.ids >= 0
    plt.plot(rad[ind],vr[ind],'k.')

    
    #radii = np.linspace(1,70,500)
    #plt.plot(radii,velr_function(radii),color='red')
    #r, vr = d.evaluate_radial_velocity_viscous(1,70)
    #plt.plot(r,vr,color='green')
    plt.show()
    #exit()




    ind = s.gas.ids < -2
    plt.plot(rad[ind],mdot[ind] ,'g.')
    ind = s.gas.ids == -2
    plt.plot(rad[ind],mdot[ind],'r.')
    ind = s.gas.ids == -1
    plt.plot(rad[ind],mdot[ind],'b.')
    ind = s.gas.ids >= 0
    plt.plot(rad[ind],mdot[ind],'k.')
    plt.ylim(0.5,1.5)
    plt.show()
    exit()


    
    #plt.plot(rad[ind],vr[ind] * s.gas.dens[ind] * rad[ind],'b.')
    #plt.plot(rad[ind],vphi[ind],'b.')
    #plt.show()


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

