"""
Script to generate 3D initial conditions of self-gravitating gas
disks around a point mass.

"""

import numpy as np
import matplotlib.pyplot as plt
import disk_models as dm
import sys
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from scipy.interpolate import interp1d

if __name__=="__main__":

    theta_incl= float(sys.argv[1]) # disk inclination in degrees
    Q = float(sys.argv[2]) # quadrupole strength at R=1
    
    Ncells = 1500000
    Mdisk = 0.05
    h0 = 0.1 # disk aspect ratio at R=1
    l = 1.0 # temperature profile index
    Rc = 16 # disk characteristic radius
    Rhole =  1.0 #disk physical inner radius
    BoxSize = 130

    quadrupole_correction = Q * 0.5 * (3 * np.cos(theta_incl*np.pi/180)**2 -1)


    
    #DISK MODEL
    '''
    sigma0,gamma = 1.0, 0.5
    def sigma_model(R):
        return np.maximum(sigma0* (1 - np.sqrt(Rhole/R)) * (R/Rc)**(-gamma) * np.exp(-(R/Rc)**(2.0-gamma)),0.0)
    '''
    Rin, Rout = 0.1,62.0
    sigma_data = np.loadtxt('viscous_solution.txt')
    x0, y0 = sigma_data[:,0],sigma_data[:,1]
    if (Rin < x0[0]):
        dx = x0[1] - x0[0]
        x0 = np.append(np.logspace(np.log10(Rin),np.log10(x0[0]-dx),10),x0)
        y0 = np.append(np.zeros(10),y0)

    sigma_model = interp1d(x0,y0)
    sigma0 = 1.0
    # Reescale the disk surface density to guarantee desired total mass.
    def mass_integrand(R):
        return 2*np.pi * R * sigma_model(R)
    Rin, Rout = 0.3,62.0
    y0 *= Mdisk/quad(mass_integrand,Rin,Rout)[0]
    sigma_model = interp1d(x0,y0,fill_value='extrapolate')

    d = dm.disk3d(sigma_function = sigma_model,
                  #sigma_type="similarity_zerotorque",
                  sigma0 = sigma0,
                  R0 = 1.0, csndR0 = 1.0,
                  csnd0=h0,
                  #l=l, Rc=Rc, R_cav = Rhole,gamma=0.5,xi=3.5,
                  l=l, Rc=Rc, Rin = 1.0,gamma=0.5,#sigma_soft=Rhole,
                  adiabatic_gamma=1.001,
                  #adiabatic_gamma =CircumstellarTempProfileIndex 1.4,
                  self_gravity = False,
                  central_particle = False,
                  quadrupole_correction = quadrupole_correction,
                  Mcentral_soft = 1.0/2.8,
                  sigma_cut = 1.0e-11)
    
    # Reescale the disk surface density to guarantee desired total mass.
    mdisk0=d.compute_disk_mass(Rin,Rout)
    print "Disk mass: %.3f" % mdisk0

    
    # Make some illustrative plots
    R,Sigma = d.evaluate_sigma(Rin,Rout)
    #refSigma = dm.similarity_sigma(R,d.sigma_disk.sigma0,
    #                               d.sigma_disk.gamma,d.sigma_disk.Rc)
    _,cs = d.evaluate_soundspeed(Rin,Rout)
    _,QT = d.evaluate_toomreQ(Rin,Rout)
    _,dmass = d.evaluate_enclosed_mass(Rin,Rout)
    plt.plot(R,np.array(dmass)*1.0/mdisk0)
    plt.plot(R,Sigma,color='orange')
    #plt.plot(R,refSigma,color='orange',ls='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
  

    # DISK MESH
    mesh = dm.disk_mesh3d(mesh_type="mc",Ncells=Ncells,fill_background=True,
                          fill_center=True,fill_box=True,BoxSize=BoxSize,
                          Rin=Rin,Rout=Rout)

    # Create SNAPSHOT
    s = dm.snapshot()
    s.create(d,mesh)
    s.incline(theta_incl,90.0,mesh)

    ind = s.gas.dens > 1e-6
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(s.gas.pos[ind,0],s.gas.pos[ind,1],s.gas.pos[ind,2],'b.',ms=0.2)
    ax.plot(s.gas.pos[np.invert(ind),0],s.gas.pos[np.invert(ind),1],
             s.gas.pos[np.invert(ind),2],'g.',ms=0.07)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.set_aspect('equal','box')
    plt.show()
    
    # Write files
    s.write_snapshot(d,mesh,relax_density_in_input = True,
                     filename='disk_Q%.2f_I%i.dat.hdf5' % (Q,theta_incl))
    s.write_parameter_file(d,mesh)

    ##########################################
    # Check tilt variables
    X0,Y0,Z0 = BoxSize * 0.5, BoxSize * 0.5,  BoxSize * 0.5
    mean_mass = Mdisk/Ncells
    print dir(s.gas)
    r = np.sqrt((s.gas.pos[:,0] - X0)**2 + 
                (s.gas.pos[:,1] - Y0)**2 +
                (s.gas.pos[:,2] - Z0)**2)
    lx = mean_mass * ((s.gas.pos[:,1] - Y0) * s.gas.vel[:,2] -
                          (s.gas.pos[:,2] - Z0) * s.gas.vel[:,1])
    ly = mean_mass * ((s.gas.pos[:,2] - Z0) * s.gas.vel[:,0] -
                          (s.gas.pos[:,0] - X0) * s.gas.vel[:,2])
    lz = mean_mass * ((s.gas.pos[:,0] - X0) * s.gas.vel[:,1] -
                          (s.gas.pos[:,1] - Y0) * s.gas.vel[:,0])
    
    # Define spherical shels
    Rin, Rout = 0.1, 15
    radii = np.logspace(np.log10(Rin),np.log10(Rout),100)
    digitized = np.digitize(r, radii)
    bin_rad = np.array([r[digitized == i].mean() for i in range(1, len(radii))])
    bin_lx = np.array([lx[digitized == i].sum() for i in range(1, len(radii))])
    bin_ly = np.array([ly[digitized == i].sum() for i in range(1, len(radii))])
    bin_lz = np.array([lz[digitized == i].sum() for i in range(1, len(radii))])
    bin_l = np.sqrt(bin_lx**2 + bin_ly**2 + bin_lz**2)
    index = bin_l != 0
    bin_rad = bin_rad[index]
    bin_lx = bin_lx[index]/bin_l[index]
    bin_ly = bin_ly[index]/bin_l[index]
    bin_lz = bin_lz[index]/bin_l[index]

    tilt = np.arctan2(np.sqrt(bin_lx**2 + bin_ly**2),bin_lz)
    twist = np.arctan2(bin_lx,-bin_ly)

    plt.plot(bin_rad,tilt*180.0/np.pi)
    plt.plot(bin_rad,twist*180.0/np.pi)
    plt.show()
