from __future__ import print_function
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
R_out = 100 #70.0
Rbreak = 70
NR = 600 #600
Nphi= 400
BOX = 220.0

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
    NR1 = NR
    NR2 = int((np.log10(Rout)-np.log10(Rbreak))/
               (np.log10(Rbreak)-np.log10(Rin)) * NR / 1.5)
    print(NR1,NR2)
    print("haha")
    Nphi1 = Nphi
    Nphi2 = int(Nphi/1.5)

    mesh = dm.disk_mesh2d(mesh_type="polar",Rin= Rin,Rout=Rout, Rbreak = Rbreak,
                          mesh_alignment = "interleaved",
                          NR1=NR1,Nphi1=Nphi1,
                          NR2=NR2,Nphi2=Nphi2,
                          Nphi_inner_bound = Nphi1,
                          Nphi_outer_bound = Nphi2,
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
    s.params.read('param_example.txt')
    s.params.reference_gas_part_mass= np.around(0.007 * sigma0 * 4*np.pi**2 * 1**2/Nphi1**2,decimals=6)
    s.params.min_volume = np.around(0.008 * np.pi * float(s.params.circumstellar_sink_radius)**2,decimals=8)
    s.params.max_volume = 20 * np.around(5*np.pi**2 * 1**2/Nphi1**2,decimals=6)
    s.params.max_volume_diff = 8
    s.params.time_limit_cpu=28000 
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

    print("Simulation parameters")
    print("alpha=",alpha)
    print("h0=",h0)
    print("qb=",qb)
    print("eb=",eb)
    print("InnerRadius=",Rin)
    print("Outer density",sigma_model(Rout))
    print("Number of cells:",s.gas.pos.shape[0])
    print("Target mass", s.params.reference_gas_part_mass)


    rad = np.sqrt((s.gas.pos[:,0]-mesh.BoxSize*0.5)**2 + (s.gas.pos[:,1]-mesh.BoxSize*0.5)**2)
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

