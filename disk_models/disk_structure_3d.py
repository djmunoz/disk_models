import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from scipy.spatial import Voronoi
import scipy.integrate as integ

from disk_density_profiles import *
from disk_external_potentials import *
from disk_other_functions import *
from disk_snapshot import *


rd.seed(42)


def SplineProfile(R,h):
  r2 = R*R
  h_inv = 1.0 / h
  h3_inv = h_inv * h_inv * h_inv
  u = R * h_inv
  wp = -1.0 / R

  ind = u < 0.5
  wp[ind] = h_inv * (-2.8 + u[ind] * u[ind] * (5.333333333333 + u[ind] * u[ind] * (6.4 * u[ind] - 9.6)))
  ind = (u >= 0.5) & (u < 1)
  wp[ind] = h_inv * (-3.2 + 0.066666666667 / u[ind] + u[ind] * u[ind] * (10.666666666667 + u[ind] * (-16.0 + u[ind] * (9.6 - 2.133333333333 * u[ind]))))
      
  return -wp
  
def SplineDerivative(R,h):
  r2 = R * R
  fac = 0.0
  h_inv = 1.0 / h
  h3_inv = h_inv * h_inv * h_inv
  u = R * h_inv
  
  fac = 1.0 / (r2 * R)
  ind = u < 0.5
  fac[ind] = h3_inv * (10.666666666667 + u[ind] * u[ind] * (32.0 * u[ind] - 38.4))
  ind = (u >= 0.5) & (u < 1)
  fac[ind] = h3_inv * (21.333333333333 - 48.0 * u[ind] + 38.4 * u[ind] * u[ind] - 10.666666666667 * u[ind] * u[ind] * u[ind] - 0.066666666667 / \
                  (u[ind] * u[ind] * u[ind]))
      
  return fac

class disk3d(object):
    def __init__(self, *args, **kwargs):
        #units
        self.G =  kwargs.get("G")
        
        #define the properties of the axi-symmetric disk model
        self.sigma_type = kwargs.get("sigma_type")
        self.sigma_disk = None
        self.sigma_cut = kwargs.get("sigma_cut")


        #Temperature profile properties
        self.csndR0 = kwargs.get("csndR0") #reference radius
        self.csnd0 = kwargs.get("csnd0") # soundspeed scaling
        self.l = kwargs.get("l") # temperature profile index

        #thermodynamic parameters
        self.adiabatic_gamma = kwargs.get("adiabatic_gamma")
        self.effective_gamma = kwargs.get("effective_gamma")        
        
        #viscosity
        self.alphacoeff = kwargs.get("alphacoeff")   

        #central object
        self.Mcentral = kwargs.get("Mcentral")
        self.Mcentral_soft = kwargs.get("Mcentral_soft")
        self.quadrupole_correction =  kwargs.get("quadrupole_correction")
        # potential type
        self.potential_type = kwargs.get("potential_type")
        
        # other properties
        self.self_gravity = kwargs.get("self_gravity")
        self.central_particle = kwargs.get("central_particle")

        #set defaults
        if (self.G is None):
            self.G = 1.0
        if (self.sigma_type is None):
            self.sigma_type="powerlaw"
        if (self.l is None):
            self.l = 1.0
        if (self.csnd0 is None):
            self.csnd0 = 0.05
        if (self.adiabatic_gamma is None):
            self.adiabatic_gamma = 7.0/5
        if (self.effective_gamma is None):
            self.effective_gamma = 1.0
        if (self.alphacoeff is None):
            self.alphacoeff = 0.01
        if (self.Mcentral is None):
            self.Mcentral = 1.0
        if (self.Mcentral_soft is None):
            self.Mcentral_soft = 0.01
        if (self.quadrupole_correction is None):
            self.quadrupole_correction = 0
        if (self.potential_type is None):
            self.potential_type="keplerian"
            
        if (self.sigma_type == "powerlaw"):
            self.sigma_disk = powerlaw_disk(**kwargs)
            if (self.csndR0 is None):
                self.csndR0 = self.sigma_disk.R0

        if (self.sigma_type == "similarity"):
            self.sigma_disk = similarity_disk(**kwargs)
            if (self.csndR0 is None):
                self.csndR0 = self.sigma_disk.Rc

        if (self.sigma_type == "powerlaw_cavity"):
            self.sigma_disk = powerlaw_cavity_disk(**kwargs)
            if (self.csndR0 is None):
                self.csndR0 = self.sigma_disk.R_cav

        if (self.sigma_type == "similarity_cavity"):
            self.sigma_disk = similarity_cavity_disk(**kwargs)
            if (self.csndR0 is None):
                self.csndR0 = self.sigma_disk.Rc

                
        if (self.sigma_cut is None):
            self.sigma_cut = self.sigma_disk.sigma0 * 1e-7

        if (self.self_gravity is None):
            self.self_gravity = False

        if (self.central_particle is None):
            self.central_particle = False

            
    def evaluate_sigma(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        sigma = self.sigma_disk.evaluate(rvals)
        sigma[sigma < self.sigma_cut] = self.sigma_cut
        return rvals,sigma

    def evaluate_enclosed_mass(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        def mass_integrand(R):
            sigma = self.sigma_disk.evaluate(R)
            if (sigma < self.sigma_cut) : sigma = self.sigma_cut
            return sigma * R * 2 * np.pi
        mass = [quad(mass_integrand,0.0,R)[0] for R in rvals]
        return rvals, mass

    def compute_disk_mass(self,Rin,Rout):
        __, mass =  self.evaluate_enclosed_mass(Rin,Rout,Nvals=2)
        return mass[1]
        
    
    def evaluate_soundspeed(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        return rvals,soundspeed(rvals,self.csnd0,self.l,self.csndR0)

    def evaluate_pressure_2d(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        return rvals, self.evaluate_sigma(Rin,Rout,Nvals,scale=scale)[1]**(self.effective_gamma) * \
            self.evaluate_soundspeed(Rin,Rout,Nvals,scale=scale)[1]**2

    def evaluate_viscosity(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals,csnd =  self.evaluate_soundspeed(Rin,Rout,Nvals,scale=scale)
        Omega_sq = self.Mcentral/rvals**3 * (1 + 3 * self.quadrupole_correction/rvals**2)
        nu = self.alphacoeff * csnd * csnd / np.sqrt(Omega_sq)
        return rvals, nu
    
    def evaluate_pressure_gradient(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals, press = self.evaluate_pressure_2d(Rin,Rout,Nvals,scale=scale)
        _, dPdR = self.evaluate_radial_gradient(press,Rin,Rout,Nvals,scale=scale)
        return rvals,dPdR

    def evaluate_angular_freq_central_gravity(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        Omega_sq = self.Mcentral * SplineDerivative(rvals,self.Mcentral_soft) * (1 + 3 * self.quadrupole_correction/rvals**2)
        return rvals, Omega_sq

    def evaluate_angular_freq_external_gravity(self,Rin,Rout,Nvals=1000,scale='log'):
        if (self.potential_type == "keplerian"):
            return self.evaluate_angular_freq_central_gravity(Rin,Rout,Nvals,scale)
        
    
    def evaluate_angular_freq_gravity(self,Rin,Rout,Nvals=1000,scale='log'):
        return self.evaluate_angular_freq_external_gravity(Rin,Rout,Nvals,scale)
        
    
    def evaluate_rotation_curve_2d(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals, Omega_sq  = self.evaluate_angular_freq_gravity(Rin,Rout,Nvals,scale)
        
        return rvals, Omega_sq + self.evaluate_pressure_gradient(Rin,Rout,Nvals,scale=scale)[1] / \
            self.evaluate_sigma(Rin,Rout,Nvals,scale=scale)[1]/ rvals

    def evaluate_toomreQ(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals, csnd = self.evaluate_soundspeed(Rin,Rout,Nvals,scale)
        _, Omega_sq = self.evaluate_angular_freq_external_gravity(Rin,Rout,Nvals,scale)
        _, Sigma = self.evaluate_sigma(Rin,Rout,Nvals,scale)

        return rvals, csnd * np.sqrt(Omega_sq) / Sigma / np.pi / self.G
        
    def evaluate_radial_velocity(self,Rin,Rout,Nvals=1000,scale='log'):
        return self.evaluate_radial_velocity_viscous(Rin,Rout,Nvals,scale=scale)

    def evaluate_radial_velocity_viscous(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        Omega = np.sqrt(self.Mcentral * SplineDerivative(rvals,self.Mcentral_soft) * (1 + 3 * self.quadrupole_correction/rvals**2))
        sigma = (self.evaluate_sigma(Rin,Rout,Nvals,scale)[1])
        _, dOmegadR = self.evaluate_radial_gradient(Omega,Rin,Rout,Nvals,scale=scale)
        
        func1 = (self.evaluate_viscosity(Rin,Rout,Nvals,scale)[1])*\
            (self.evaluate_sigma(Rin,Rout,Nvals,scale)[1])*\
            rvals**3*dOmegadR
        _, dfunc1dR = self.evaluate_radial_gradient(func1,Rin,Rout,Nvals,scale=scale)
        func2 = rvals**2 * Omega
        _, dfunc2dR = self.evaluate_radial_gradient(func2,Rin,Rout,Nvals,scale=scale)

        velr = dfunc1dR / rvals / sigma / dfunc2dR
        velr[sigma <= self.sigma_cut] = 0
        

        return rvals,velr

    def evaluate_radial_velocity_constant_mdot(self,Rin,Rout,Nvals=1000,scale='log'):
        
        return 0
        
    def evaluate_radial_gradient(self,quantity,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        if (scale == 'log'):
            dQdlogR = np.gradient(quantity)/np.gradient(np.log10(rvals))
            dQdR = dQdlogR/rvals/np.log(10)
        elif (scale == 'linear'):
            dQdR = np.gradient(quantity)/np.gradient(rvals)
        return rvals,dQdR

    def evaluate_radial_zones(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", scale, "not known!"
            sys.exit()
        return rvals

    def evaluate_radial_mass_bins(self,Rin,Rout,Nbins):
        rvals,mvals = self.evaluate_enclosed_mass(Rin,Rout,Nvals=2000)
        MasstoRadius=interp1d(np.append([0],mvals),np.append([0],rvals),kind='linear')
        radial_bins= MasstoRadius(np.linspace(0.0,1.0,Nbins)*max(mvals))
        return radial_bins

    def evaluate_vertical_structure_selfgravity(self,R,zin,zout,Nzvals=400,G=1):
        zvals = np.logspace(np.log10(zin),np.log10(zout),Nzvals)

        m_enclosed =  self.sigma_disk.evaluate(R) * np.pi * R**2
        
        # First take a guess of the vertical structure        
        if (m_enclosed < 0.1 * self.Mcentral):
            zrho0=[np.exp(-self.vertical_potential(R,zz)/soundspeed(R,self.csnd0,self.l,self.csndR0)**2) for zz in zvals]
            VertProfileNorm = self.sigma_disk.evaluate(R)/(2.0*cumtrapz(zrho0,zvals))[-1]
        else:
            VertProfileNorm = self.sigma_disk.evaluate(R)**2/soundspeed(R,self.csnd0,self.l,self.csndR0)**2/ 2.0 * np.pi * G

        VertProfileNorm_old = 1.0e-40
        if (VertProfileNorm < VertProfileNorm_old): VertProfileNorm = VertProfileNorm_old


        
        def VerticalPotentialEq(Phi,z,r,rho0):
            dPhiGrad_dz =  rho0 * G* 4 * np.pi * np.exp(-(self.vertical_potential(r,z) + Phi[1]) / soundspeed(r,self.csnd0,self.l,self.csndR0)**2)
            dPhi_dz = Phi[0]
            return [dPhiGrad_dz,dPhi_dz]

        iterate = 0
        while(True) : #iteration
            min_step = (zvals[1]-zvals[0])/100.0
            tol = 1.0e-9
            # Solve for the vertical potential
            soln=integ.odeint(VerticalPotentialEq,[0.0,0.0],zvals,args=(R,VertProfileNorm),rtol=tol,
                              mxords=15,hmin=min_step,printmessg=False)[:,1]
            zrho0=[np.exp(-(self.vertical_potential(R,zvals[kk])+soln[kk])/soundspeed(R,self.csnd0,self.l,self.csndR0)**2) for kk in range(len(zvals))]
            VertProfileNorm = self.sigma_disk.evaluate(R)/(2.0*cumtrapz(zrho0,zvals))[-1]
            zrho = [VertProfileNorm * r for r in zrho0]

            if (VertProfileNorm * 1.333 * np.pi * R**3 < 1.0e-12 * self.Mcentral): break
            # Check if vertical structure solution has converged
            abstol,reltol = 1.0e-9,1.0e-6
            abserr,relerr = np.abs(VertProfileNorm_old-VertProfileNorm),np.abs(VertProfileNorm_old-VertProfileNorm)/np.abs(VertProfileNorm_old)
            
            if (np.abs(VertProfileNorm) > abstol/reltol):
                if (abserr < abstol): break
            else:
                if (relerr < reltol): break
            VertProfileNorm_old = VertProfileNorm
            iterate += 1

        return zvals,zrho,VertProfileNorm

    def evaluate_vertical_structure_no_selfgravity(self,R,zin,zout,Nzvals=400):
        zvals = np.logspace(np.log10(zin),np.log10(zout),Nzvals)
        #def integrand(z): return np.exp(-self.vertical_potential(R,z)/soundspeed(R,self.csnd0,self.l,self.csndR0)**2)
        #VertProfileNorm =  self.sigma_disk.evaluate(R)/(2.0*quad(integrand,0,zout*15)[0])
        zrho0=[np.exp(-self.vertical_potential(R,zz)/soundspeed(R,self.csnd0,self.l,self.csndR0)**2) for zz in zvals]
        VertProfileNorm = self.sigma_disk.evaluate(R)/(2.0*cumtrapz(zrho0,zvals))[-1]
        #zrho = [VertProfileNorm * np.exp(-self.vertical_potential(R,zz)/soundspeed(R,self.csnd0,self.l,self.csndR0)**2) for zz in zvals]
        zrho = [VertProfileNorm * zrho0[kk] for kk in range(len(zrho0))]
        return zvals,zrho,VertProfileNorm

    def evaluate_vertical_structure(self,R,zin,zout,Nzvals=400):
        if (self.self_gravity):
            return self.evaluate_vertical_structure_selfgravity(R,zin,zout,Nzvals)
        else:
            return self.evaluate_vertical_structure_no_selfgravity(R,zin,zout,Nzvals)
        
    def evaluate_enclosed_vertical(self,R,zin,zout,Nzvals=400):
        zvals, zrho, _ = self.evaluate_vertical_structure(R,zin,zout,Nzvals)
        zmass = np.append(0,cumtrapz(zrho,zvals))
        return zvals, zmass
    
    def spherical_potential(self,r):
        if (self.potential_type == "keplerian"):
            return -self.G * self.Mcentral * spherical_potential_keplerian(r,self.Mcentral_soft)
    
    def vertical_potential(self,R,z):
        return (self.spherical_potential(np.sqrt(R*R + z*z)) - self.spherical_potential(R))

    def solve_vertical_structure(self,Rsamples,zsamples,Rin,Rout,Ncells):

        dens = np.zeros(Rsamples.shape[0])
        
        if (Ncells < 50000): R_bins = 160
        elif (Ncells < 100000): R_bins = 200
        elif (Ncells < 200000): R_bins = 300
        elif (Ncells < 600000): R_bins   = 400
        else: R_bins = 500
        radial_bins = self.evaluate_radial_mass_bins(Rin,Rout,R_bins)
        #fix the bins a bit
        dRin = radial_bins[1]-radial_bins[0]
        radial_bins = np.append(0,np.append(np.arange(radial_bins[1]/30,radial_bins[1],dRin/30),radial_bins[1:]))
        
        bin_inds=np.digitize(Rsamples,radial_bins)
        mid_plane = []
        radii = []
        zin,zout = 0.99*np.abs(zsamples).min(),1.01*np.abs(zsamples).max()

        print "Solving vertical structure AGAIN for density evaluation at the sampled locations"
        for kk in range(0,radial_bins.shape[0]):
            update_progress(kk,radial_bins.shape[0])
            N_in_bin = Rsamples[bin_inds == kk].shape[0]
            if (N_in_bin == 0):
                mid_plane.append(0.0)
                radii.append(radial_bins[kk])
                continue
            #if (N_in_bin < 10) & (zout > 5*(z[bin_inds == kk]).mean()): #| (zout/200 > np.abs(z[bin_inds == kk]).max()):
            #    zout = np.abs(z[bin_inds == kk]).mean()*5
            bin_radius = Rsamples[bin_inds == kk].mean()
            zvals,zrhovals,rho0 = self.evaluate_vertical_structure(bin_radius,zin,zout,Nzvals=800)
            mid_plane.append(rho0)
            radii.append(bin_radius)
            dens_profile = interp1d(zvals,zrhovals,kind='linear')
            dens[bin_inds == kk] = dens_profile(np.abs(zsamples[bin_inds == kk]))

        return dens,np.array(radii),np.array(mid_plane)
        
    
class disk_mesh3d():
    def __init__(self, *args, **kwargs):

        self.mesh_type=kwargs.get("mesh_type")
        self.Rin = kwargs.get("Rin")
        self.Rout = kwargs.get("Rout")
        self.zmax = 0.0
        self.NR = kwargs.get("NR")
        self.Nphi = kwargs.get("Nphi")
        self.Ncells = kwargs.get("Ncells")
        self.BoxSize = kwargs.get("BoxSize")
        self.mesh_alignment = kwargs.get("mesh_alignment")
        self.N_inner_boundary_rings = kwargs.get("N_inner_boundary_rings")
        self.N_outer_boundary_rings = kwargs.get("N_outer_boundary_rings") 
        self.fill_box = kwargs.get("fill_box")
        self.fill_center = kwargs.get("fill_center")
        self.fill_background = kwargs.get("fill_background")
        self.max_fill_mesh_points =  kwargs.get("max_fill_mesh_points")
        
        # set default values
        if (self.mesh_type is None):
            self.mesh_type="polar"
        if (self.Rin is None):
            self.Rin = 1
        if (self.Rout is None):
            self.Rout = 10
        if (self.NR is None):
            self.NR = 800
        if (self.Nphi is None):
            self.Nphi = 600
        if (self.Ncells is None):
            self.Ncells =  self.NR * self.Nphi
        if (self.N_inner_boundary_rings is None):
            self.N_inner_boundary_rings = 1
        if (self.N_outer_boundary_rings is None):
            self.N_outer_boundary_rings = 1            
            
        if (self.BoxSize is None):
            self.BoxSize = 1.2 * 2* self.Rout
            
        if (self.fill_box is None):
            self.fill_box = False
        if (self.fill_center is None):
            self.fill_center = False   
        if (self.fill_background is None):
            self.fill_background = False
        if (self.max_fill_mesh_points is None):
            self.max_fill_mesh_points = 0.15 * self.Ncells

            
    def create(self,disk):

        
        if (self.mesh_type == "polar"):

            rvals = np.logspace(np.log10(self.Rin),np.log10(self.Rout),self.NR+1)
            rvals = rvals[:-1] + 0.5 * np.diff(rvals)
            self.deltaRin,self.deltaRout = rvals[1]-rvals[0],rvals[-1]-rvals[-2]
            for kk in range(self.N_inner_boundary_rings): rvals=np.append(rvals[0]-self.deltaRin, rvals)
            for kk in range(self.N_outer_boundary_rings): rvals=np.append(rvals,rvals[-1]+self.deltaRout)

            phivals = np.linspace(0,2*np.pi,self.Nphi+1)
            R,phi = np.meshgrid(rvals,phivals)
            
            if (self.mesh_alignment == "interleaved"):
                phi[:-1,2*self.N_inner_boundary_rings:-2*self.N_outer_boundary_rings:2] = phi[:-1,2*self.N_inner_boundary_rings:-2*self.N_outer_boundary_rings:2] + 0.5*np.diff(phi[:,2*self.N_inner_boundary_rings:-2*self.N_outer_boundary_rings:2],axis=0)
                
            phi = phi[:-1,:]
            R = R[:-1,:]
            R, phi = R.flatten(),phi.flatten()
            
            if (self.fill_box == True):
                rvals = np.array([R.max()+self.deltaRout,R.max()+2* self.deltaRout])
                phivals = np.arange(0,2*np.pi,2*np.pi/(0.5*self.Nphi))
                Rback,phiback = np.meshgrid(rvals,phivals)
                R = np.append(R,Rback.flatten())
                phi = np.append(phi,phiback.flatten())

                extent = 0.5 * self.BoxSize - 2*self.deltaRout
                interval = 4*self.deltaRout
                xback,yback = np.meshgrid(np.arange(-extent + 0.5 * interval, extent,interval),
                                          np.arange(-extent + 0.5 * interval, extent,interval))
                xback,yback = xback.flatten(),yback.flatten()
                Rback = np.sqrt(xback**2+yback**2)
                phiback = np.arctan2(yback,xback)
                ind = Rback > R.max()+2.5 * self.deltaRout
                Rback, phiback = Rback[ind], phiback[ind]

                print "....adding %i additional mesh-generating points" % (Rback.shape[0])
                R = np.append(R,Rback)
                phi = np.append(phi,phiback)

            if (self.fill_center == True):
                rvals = np.array([R.min()-3* self.deltaRin,R.min()-self.deltaRin])
                phivals = np.arange(0,2*np.pi,2*np.pi/(0.5*self.Nphi))
                Rcenter,phicenter = np.meshgrid(rvals,phivals)
                R = np.append(R,Rcenter.flatten())
                phi = np.append(phi,phicenter.flatten())

                extent = self.Rin 
                interval = 3* self.deltaRin
                xcenter,ycenter= np.meshgrid(np.arange(-extent + 0.5 * interval, extent,interval),
                                          np.arange(-extent + 0.5 * interval, extent,interval))
                xcenter,ycenter = xcenter.flatten(),ycenter.flatten()
                Rcenter = np.sqrt(xcenter**2+ycenter**2)
                phicenter = np.arctan2(ycenter,xcenter)
                ind = Rcenter < R.min() - 2* self.deltaRin
                Rcenter, phicenter = Rcenter[ind], phicenter[ind]

                print "....adding %i additional mesh-generating points" % (Rcenter.shape[0])
                R = np.append(R,Rcenter)
                phi = np.append(phi,phicenter)

            z = np.zeros(R.shape[0])
                
            return R,phi,z

        if (self.mesh_type == "mc"):
            R,phi = self.mc_sample_2d(disk)
            z = self.mc_sample_vertical(R,disk)

            if (self.fill_background | self.fill_center | self.fill_box):
                Radditional, phiadditional, zadditional = np.empty([0]),np.empty([0]),np.empty([0])

            self.zmax = np.abs(z).max()
            zmax  = np.abs(z).max()
            Rmin  = R.min()
            Rmax  = R.max()
            
            if (self.fill_background == True):
                Rback,phiback = self.mc_sample_2d(disk,Npoints=0.1 * self.Ncells)
                zback = self.mc_sample_vertical_background(R,Rback,z,disk)
                Rbackmax = Rback.max()
                print "....adding %i additional mesh-generating points" % (Rback.shape[0])
                Radditional = np.append(Radditional,Rback).flatten()
                phiadditional = np.append(phiadditional,phiback).flatten()
                zadditional = np.append(zadditional,zback).flatten()

                Lx,Ly,Lz = 2*Rbackmax,2*Rbackmax,np.abs(zback).max()+1.2*(Rbackmax -Rmax)
                delta = zback.max()/3
                xback,yback,zback = self.sample_fill_box(0,Lx,0,Ly,0,Lz,delta)
                Rback = np.sqrt(xback**2+yback**2)
                phiback = np.arctan2(yback,xback)
                ind = Rback > Rbackmax
                Rback, phiback,zback = Rback[ind], phiback[ind],zback[ind]

                print "....adding %i additional mesh-generating points" % (Rback.shape[0])
                Radditional = np.append(Radditional,Rback)
                phiadditional = np.append(phiadditional,phiback)
                zadditional = np.append(zadditional,zback)

                zmax = max(zmax,np.abs(zadditional).max())
                Rmax = max(Rmax,np.abs(Radditional).max())
                Rmin = min(Rmin,np.abs(Radditional).min())

            
            if (self.fill_center == True):
                rvals,mvals = disk.evaluate_enclosed_mass(self.Rin, self.Rout,Nvals=100)
                cellmass = mvals[-1]/self.Ncells
                m2r=interp1d(np.append([0],mvals),np.append([0],rvals),kind='linear')
                Rmin = m2r(cellmass)
                sigma_in = disk.sigma_disk.evaluate(Rmin)
                if (sigma_in < disk.sigma_cut): sigma_in = disk.sigma_cut
                rho_in = sigma_in/zmax
                delta = 0.15*(cellmass/rho_in)**0.333333
                Lx, Ly, Lz = 2 * Rmin, 2 * Rmin,2*zmax
                xcenter,ycenter,zcenter =  self.sample_fill_box(0,Lx,0,Ly,0,Lz,delta)
                Rcenter = np.sqrt(xcenter**2+ycenter**2)
                phicenter = np.arctan2(ycenter,xcenter)
                ind = Rcenter < Rmin 
                Rcenter, phicenter,zcenter = Rcenter[ind], phicenter[ind],zcenter[ind]

                print "....adding %i additional mesh-generating points" % (Rcenter.shape[0])
                Radditional = np.append(Radditional,Rcenter)
                phiadditional = np.append(phiadditional,phicenter)
                zadditional = np.append(zadditional,zcenter)

                zmax = max(zmax,np.abs(zadditional).max())
                Rmax = max(Rmax,np.abs(Radditional).max())
                Rmin = min(Rmin,np.abs(Radditional).min())

            
            if (self.fill_box == True):
                print "Filling computational box of side half-length",self.BoxSize/2
                zmax0 = zmax
                Rmax0 = Rmax
                Nlayers = 0
                Lx, Ly, Lz = min(2.1 * Rmax0,self.BoxSize), min(2.1 * Rmax0,self.BoxSize), min(2.1 * zmax0,self.BoxSize)
                delta  = 1.5 * (zmax0/Rmax0)* zmax0
                xbox,ybox,zbox =  self.sample_fill_box(0,Lx,0,Ly,0,Lz,delta)
                rbox = np.sqrt(xbox**2+ybox**2)
                ind = (rbox > Rmax0) | (np.abs(zbox) > zmax0)
                print "....adding %i additional mesh-generating points out to x=+-%f" % (rbox[ind].shape[0],xbox.max())
                Radditional = np.append(Radditional,rbox[ind])
                phiadditional = np.append(phiadditional,np.arctan2(ybox[ind],xbox[ind]))
                zadditional=np.append(zadditional,zbox[ind])

                delta*=1.9
                while (Lx < self.BoxSize-0.5*delta) | (Lz < self.BoxSize- 0.5*delta):
                #while ((0.5*self.BoxSize > (R.max()/np.sqrt(2) + 1.5*delta))
                #       | (0.5*self.BoxSize > (np.abs(z).max()+1.5*delta))):
                    #print Lx, Ly, Lz,R.max()/np.sqrt(2),z.max(),delta
                    if (Nlayers > 8): break
                    Nlayers+=1
                    lmax,zetamax = Radditional.max()/np.sqrt(2), zadditional.max()
                    Lx, Ly, Lz = min(1.6 * Lx,self.BoxSize), min(1.6 * Ly,self.BoxSize), min(3.8 * Lz,self.BoxSize)
                    Lx_in, Ly_in, Lz_in = np.abs(Radditional*np.cos(phiadditional)).max(),np.abs(Radditional*np.sin(phiadditional)).max(),zetamax

                    xbox,ybox,zbox =  self.sample_fill_box(Lx_in,Lx,Ly_in,Ly,Lz_in,Lz,delta)

                    print "....adding %i additional mesh-generating points out to x=+-%f" % (xbox.shape[0],xbox.max())
                    Radditional = np.append(Radditional,np.sqrt(xbox**2+ybox**2))
                    phiadditional = np.append(phiadditional,np.arctan2(ybox,xbox))
                    zadditional=np.append(zadditional,zbox)

                    delta  = min(max(Lx,Lz)*1.0/16*Nlayers,0.6*min(Lx-Lx_in,Lz-Lz_in))

            if (self.fill_background | self.fill_center | self.fill_box):

                # Check if we added TOO MANY additional mesh points
                if (Radditional.shape[0] > self.max_fill_mesh_points):
                    print "...removing excessive extra points"
                    # Randomly select a subsample of size equal to the maximum allowed size
                    ind = rd.random_sample(Radditional.shape[0]) <  self.max_fill_mesh_points/Radditional.shape[0]
                    Radditional = Radditional[ind]
                    phiadditional = phiadditional[ind]
                    zadditional = zadditional[ind]

                R = np.append(R,Radditional)
                phi = np.append(phi,phiadditional)
                z = np.append(z,zadditional)

            print "Added a total of %i extra points\n" % Radditional.shape[0]
            return R,phi,z

                             
    def mc_sample_2d(self,disk,**kwargs):

        Npoints = kwargs.get("Npoints")
        if (Npoints is None): Npoints = self.Ncells
        
        rvals,mvals = disk.evaluate_enclosed_mass(self.Rin, self.Rout)
        R = self.mc_sample_from_mass(rvals,mvals,int(Npoints))
        Rmax = R.max()
        while (R[R < Rmax].shape[0] > 1.01 * Npoints):
            R = R[R< (0.98 * Rmax)]
            Rmax = R.max()
            
        Ncells = R.shape[0]
        phi=2.0*np.pi*rd.random_sample(int(Npoints))

        return R,phi

    def mc_sample_vertical(self,R,disk):

        if (self.Ncells < 50000): R_bins = 80
        elif (self.Ncells < 100000): R_bins = 120
        elif (self.Ncells < 200000): R_bins = 200
        elif (self.Ncells < 600000): R_bins   = 400
        else: R_bins = 500
        
        #bin radial values (use mass as a guide for bin locations)
        radial_bins = disk.evaluate_radial_mass_bins(self.Rin,self.Rout,R_bins)
        bin_inds=np.digitize(R,radial_bins)
        z = np.zeros(R.shape[0])

        print "Solving vertical structure for point location sampling:"
        for kk in range(0,R_bins):
            update_progress(kk,R_bins)
            
            N_in_bin = R[bin_inds == kk].shape[0]
            if (N_in_bin == 0): continue

            if (N_in_bin < 10):
                z[bin_inds == kk] = 0.0
                continue
            
            bin_radius = R[bin_inds == kk].mean()
            scale_height_guess = bin_radius * soundspeed(bin_radius,disk.csnd0,disk.l,disk.csndR0)/np.sqrt(disk.Mcentral/bin_radius)
            zin,zout = 0.001 * scale_height_guess , 15 * scale_height_guess
            zvals,zmvals = disk.evaluate_enclosed_vertical(bin_radius,zin,zout,Nzvals=400)
            zbin = self.mc_sample_from_mass(zvals,zmvals,int(1.2*N_in_bin))
            zbinmax = zbin.max()
            while (zbin[zbin < zbinmax].shape[0] > 1.04 * N_in_bin):
                zbin = zbin[zbin< (0.99 * zbinmax)]
                zbinmax = zbin.max()
            if (zbin.shape[0] > N_in_bin): zbin = zbin[:N_in_bin]

            #points below or above the mid-plane
            zbin = zbin * (np.round(rd.random_sample(N_in_bin))*2 - 1)
            z[bin_inds == kk] = zbin

        return z

    def mc_sample_vertical_background(self,R,Rback,z,disk):

        zmax = 1.3*np.abs(z).max()
        zback = np.zeros(Rback.shape[0])
    
        if (self.Ncells < 50000): R_bins = 40
        elif (self.Ncells < 100000): R_bins = 60
        elif (self.Ncells < 200000): R_bins = 100
        elif (self.Ncells < 600000): R_bins   = 200
        else: R_bins = 250
        
        #bin radial values (use mass as a guide for bin locations)
        radial_bins = disk.evaluate_radial_mass_bins(self.Rin,self.Rout,R_bins)
        bin_inds = np.digitize(R,radial_bins)
        backbin_inds = np.digitize(Rback,radial_bins)

        for kk in range(0,R_bins):
            N_in_bin = R[bin_inds == kk].shape[0]
            Nback_in_bin = Rback[backbin_inds == kk].shape[0]
            if (N_in_bin == 0) | (Nback_in_bin == 0) : continue
            
            zbinmax = z[bin_inds == kk].max()
            bin_radius = Rback[backbin_inds == kk].mean()

            scale_height_guess = bin_radius * soundspeed(bin_radius,disk.csnd0,disk.l,disk.csndR0)/np.sqrt(disk.Mcentral/bin_radius)
            zbackbin = rd.random_sample(Nback_in_bin)*(zmax - zbinmax) + zbinmax
            zback[backbin_inds == kk] = zbackbin * (np.round(rd.random_sample(Nback_in_bin))*2 - 1)

        return zback
    
    def mc_sample_from_mass(self,x,m,N):
        #m2x=InterpolatedUnivariateSpline(m, x,k=1)
        m2x=interp1d(np.append([0],m),np.append([0],x),kind='linear')
        xran = m2x(rd.random_sample(N)*max(m))
        return xran

                
    def sample_fill_box(self,Lx_in,Lx_out,Ly_in,Ly_out,Lz_in,Lz_out,delta):
        
        xbox,ybox,zbox = np.meshgrid(np.arange(-(0.5 * Lx_out)+0.5*delta, (0.5 * Lx_out),delta),
                                     np.arange(-(0.5 * Ly_out)+0.5*delta, (0.5 * Ly_out),delta),
                                     np.arange(-(0.5 * Lz_out)+0.5*delta, (0.5 * Lz_out),delta))
        xbox,ybox,zbox =  xbox.flatten(),ybox.flatten(),zbox.flatten()
        ind = (np.abs(xbox) > 0.5* Lx_in) | (np.abs(ybox) > 0.5*Ly_in) | (np.abs(zbox) > 0.5*Lz_in) 
        
        xbox,ybox,zbox =  xbox[ind],ybox[ind],zbox[ind]
        
        return xbox,ybox,zbox



            

if __name__=="__main__":


    d = disk()
    m = disk_mesh(d)
    m.create()
    ss = snapshot()
    ss.create(d,m)
    ss.write_snapshot(d,m)
    
