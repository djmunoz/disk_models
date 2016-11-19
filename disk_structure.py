import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import snapHDF5 as ws
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from scipy.spatial import Voronoi
import scipy.integrate as integ

from disk_density_profiles import *
from disk_external_potentials import *






class disk(object):
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
            self.Mcentral_soft = 1.0
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

    def evaluate_angular_freq_centralgravity(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        Omega_sq = self.Mcentral/rvals**3 * (1 + 3 * self.quadrupole_correction/rvals**2)
        return rvals, Omega_sq
    
    def evaluate_rotation_curve_2d(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals, Omega_sq  = self.evaluate_angular_freq_centralgravity(Rin,Rout,Nvals,scale)
        
        return rvals, Omega_sq + self.evaluate_pressure_gradient(Rin,Rout,Nvals,scale=scale)[1] / \
            self.evaluate_sigma(Rin,Rout,Nvals,scale=scale)[1]/ rvals

    def evaluate_radial_velocity(self,Rin,Rout,Nvals=1000,scale='log'):
        return self.evaluate_radial_velocity_viscous(Rin,Rout,Nvals,scale=scale)

    def evaluate_radial_velocity_viscous(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        Omega = np.sqrt(self.Mcentral/rvals**3 * (1 + 3 * self.quadrupole_correction/rvals**2))
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
        rvals,mvals = self.evaluate_enclosed_mass(Rin,Rout)
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
        print VertProfileNorm_old,VertProfileNorm,m_enclosed < 0.1 * self.Mcentral,self.Mcentral,R

        
        def VerticalPotentialEq(Phi,z,r,rho0):
            dPhiGrad_dz =  rho0 * G* 4 * np.pi * np.exp(-(self.vertical_potential(r,z) + Phi[1]) / soundspeed(r,self.csnd0,self.l,self.csndR0)**2)
            dPhi_dz = Phi[0]
            return [dPhiGrad_dz,dPhi_dz]

        iterate = 0
        while(True) : #iteration
            #print iterate
            min_step = (zvals[1]-zvals[0])/100.0
            tol = 1.0e-8
            # Solve for the vertical potential
            soln=integ.odeint(VerticalPotentialEq,[0.0,0.0],zvals,args=(R,VertProfileNorm),rtol=tol,
                              mxords=15,hmin=min_step,printmessg=False)[:,1]
            zrho0=[np.exp(-(self.vertical_potential(R,zvals[kk])+soln[kk])/soundspeed(R,self.csnd0,self.l,self.csndR0)**2) for kk in range(len(zvals))]
            VertProfileNorm = self.sigma_disk.evaluate(R)/(2.0*cumtrapz(zrho0,zvals))[-1]
            zrho = [VertProfileNorm * r for r in zrho0]

            if (VertProfileNorm * 1.333 * np.pi * R**3 < 1.0e-12 * self.Mcentral): break
            # Check if vertical structure solution has converged
            abstol,reltol = 1.0e-8,1.0e-5
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
        print R,"hello"
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
        elif (Ncells < 600000): R_bins   = 500
        else: R_bins = 700
        radial_bins = self.evaluate_radial_mass_bins(Rin,Rout,R_bins)
        #fix the bins a bit
        dRin = radial_bins[1]-radial_bins[0]
        radial_bins = np.append(0,np.append(np.arange(radial_bins[1]/50,radial_bins[1],dRin/50),radial_bins[1:]))
        
        bin_inds=np.digitize(Rsamples,radial_bins)
        mid_plane = []
        radii = []
        zin,zout = 0.99*np.abs(zsamples).min(),1.01*np.abs(zsamples).max()
        for kk in range(0,radial_bins.shape[0]):
            N_in_bin = Rsamples[bin_inds == kk].shape[0]
            if (N_in_bin == 0):
                mid_plane.append(0.0)
                radii.append(radial_bins[kk])
                continue
            #if (N_in_bin < 10) & (zout > 5*(z[bin_inds == kk]).mean()): #| (zout/200 > np.abs(z[bin_inds == kk]).max()):
            #    zout = np.abs(z[bin_inds == kk]).mean()*5
            bin_radius = Rsamples[bin_inds == kk].mean()
            zvals,zrhovals,rho0 = self.evaluate_vertical_structure(bin_radius,zin,zout,Nzvals=600)
            mid_plane.append(rho0)
            radii.append(bin_radius)
            dens_profile = interp1d(zvals,zrhovals,kind='linear')
            dens[bin_inds == kk] = dens_profile(np.abs(zsamples[bin_inds == kk]))

        return dens,np.array(radii),np.array(mid_plane)
        
    
class disk_mesh():
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


            if (self.fill_background == True):
                Rback,phiback = self.mc_sample_2d(disk,Npoints=0.1 * self.Ncells)
                zback = self.mc_sample_vertical_background(R,Rback,z,disk)
                Rbackmax = Rback.max()
                print "....adding %i additional mesh-generating points" % (Rback.shape[0])
                R = np.append(R,Rback)
                phi = np.append(phi,phiback)
                z = np.append(z,zback)
                
                Lx,Ly,Lz = 2*Rbackmax,2*Rbackmax,2.2*np.abs(zback).max()
                delta = zback.max()/3
                xback,yback,zback = self.sample_fill_box(0,Lx,0,Ly,0,Lz,delta)
                Rback = np.sqrt(xback**2+yback**2)
                phiback = np.arctan2(yback,xback)
                ind = Rback > Rbackmax
                Rback, phiback,zback = Rback[ind], phiback[ind],zback[ind]

                print "....adding %i additional mesh-generating points" % (Rback.shape[0])
                R = np.append(R,Rback)
                phi = np.append(phi,phiback)
                z = np.append(z,zback)

            self.zmax = np.abs(z).max()
                
            if (self.fill_center == True):
                Rmin  = R.min()
                Rmax  = R.max()
                zmax  = np.abs(z).max()
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
                R = np.append(R,Rcenter)
                phi = np.append(phi,phicenter)
                z = np.append(z,zcenter)

            if (self.fill_box == True):
                zmax0 = np.abs(z).max()
                Rmax0 = R.max()
                Nlayers = 0
                Lx, Ly, Lz = min(2.1 * Rmax0,self.BoxSize), min(2.1 * Rmax0,self.BoxSize), min(2.1 * zmax0,self.BoxSize)
                delta  = 0.4 * zmax0
                xbox,ybox,zbox =  self.sample_fill_box(0,Lx,0,Ly,0,Lz,delta)
                rbox = np.sqrt(xbox**2+ybox**2)
                ind = (rbox > Rmax0) | (np.abs(zbox) > zmax0)
                print "....adding %i additional mesh-generating points" % (rbox[ind].shape[0])
                R = np.append(R,rbox[ind])
                phi = np.append(phi,np.arctan2(ybox[ind],xbox[ind]))
                z=np.append(z,zbox[ind])

                while (Lx < self.BoxSize-0.5*delta) | (Lz < self.BoxSize- 0.5*delta):
                #while ((0.5*self.BoxSize > (R.max()/np.sqrt(2) + 1.5*delta))
                #       | (0.5*self.BoxSize > (np.abs(z).max()+1.5*delta))):
                    #print Lx, Ly, Lz,R.max()/np.sqrt(2),z.max(),delta
                    if (Nlayers > 8): break
                    Nlayers+=1
                    lmax,zmax = R.max()/np.sqrt(2), z.max()
                    Lx, Ly, Lz = min(2.0 * Lx,self.BoxSize), min(2.0 * Ly,self.BoxSize), min(6 * Lz,self.BoxSize)
                    Lx_in, Ly_in, Lz_in = np.abs(R*np.cos(phi)).max(),np.abs(R*np.sin(phi)).max(),zmax
                    delta  = min(max(Lx,Lz)*1.0/32*Nlayers,0.4*min(Lx-Lx_in,Lz-Lz_in))
                    xbox,ybox,zbox =  self.sample_fill_box(Lx_in,Lx,Ly_in,Ly,Lz_in,Lz,delta)

                    print "....adding %i additional mesh-generating points" % (xbox.shape[0])
                    R = np.append(R,np.sqrt(xbox**2+ybox**2))
                    phi = np.append(phi,np.arctan2(ybox,xbox))
                    z=np.append(z,zbox)

            
 
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
        phi=2.0*np.pi*rd.random_sample(Ncells)

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
        for kk in range(0,R_bins):
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

        zmax = 1.1*np.abs(z).max()
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


class params():
     def __init__(self,*args,**kwargs):
        self.targetmass=kwargs.get("targetmass")
        self.minvol=kwargs.get("minvol")
        self.maxvol=kwargs.get("maxvol")

        
        if (self.targetmass is None):
            self.targetmass = 0.0
        if (self.minvol is None):
            self.minvol = 0.0
        if (self.maxvol is None):
            self.maxvol = 0.0


            
class snapshot():

    def __init__(self,*args,**kwargs):
        self.pos=kwargs.get("pos")
        self.vel=kwargs.get("vel")
        self.dens=kwargs.get("dens")
        self.utherm=kwargs.get("utherm")
        self.ids=kwargs.get("ids")

        self.params = params()
        
    def create(self,disk,disk_mesh):
        
        R,phi,z,dens,vphi,vr,press,ids = self.assign_primitive_variables(disk,disk_mesh)


        print "right here?"
        self.load(R,phi,z,dens,vphi,vr,press,ids,disk_mesh.BoxSize,disk.adiabatic_gamma)

        rvals,mvals = disk.evaluate_enclosed_mass(disk_mesh.Rin,disk_mesh.Rout,Nvals=100)
        self.params.targetmass = mvals[-1]/disk_mesh.Ncells
        ind = (R < 1.2 * disk_mesh.Rout) & ((R > disk_mesh.Rout))
        self.params.maxvol = 4.0/3*np.pi * disk_mesh.Rout**3 * (1.2**3-1.0)/ R[ind].shape[0]
        ind = (R > disk_mesh.Rin) & ((R < disk_mesh.Rout))
        self.params.maxvol = self.params.targetmass/dens[ind].min()
        self.params.minvol = self.params.targetmass/dens[ind].max()

        print "VOL",self.params.minvol,self.params.maxvol  
        

        
        print disk_mesh.Rin
        
    def load(self,R,phi,z,dens,vphi,vr,press,ids,BoxSize,adiabatic_gamma):
        
        x = R * np.cos(phi) + 0.5 * BoxSize
        y = R * np.sin(phi) + 0.5 * BoxSize
        z = z + 0.5 * BoxSize
        
        vx = vr * np.cos(phi) - vphi * np.sin(phi)
        vy = vr * np.sin(phi) + vphi * np.cos(phi)
        vz = np.zeros(vx.shape[0])
        
        self.dens = dens
        self.pos = np.array([x,y,z]).T
        self.vel = np.array([vx,vy,vz]).T
        print "GAMMA", adiabatic_gamma
        self.utherm = press/self.dens/(adiabatic_gamma - 1)
        self.ids = ids
    
    
     
    def assign_primitive_variables(self,disk,disk_mesh):
        
        R,phi,z = disk_mesh.create(disk)

        x = R*np.cos(phi)
        y = R*np.sin(phi)
        points = np.array([x,y,z]).T
        print points.shape
        #print "Voronoi"
        #vor = Voronoi(points)
        
        R1,R2 = 1e-4,1.5*disk_mesh.Rout
        #obtain density of cells
        dens, radii, midplane_dens = disk.solve_vertical_structure(R,z,R1,R2,disk_mesh.Ncells)
        dens_cut = midplane_dens[-1]
        dens[dens < dens_cut] = dens_cut
        midplane_dens[midplane_dens < dens_cut] = dens_cut
        dens0_profile =  interp1d(radii,midplane_dens,kind='linear')

        rr,sig =  disk.evaluate_sigma(R1,R2)
        _,cs0 =  disk.evaluate_soundspeed(R1,R2)
        dens0_guess = sig/np.sqrt(2*np.pi)/(cs0*np.sqrt(rr)*rr)

        
        #evaluate other quantities
        R1,R2 = 0.99*R.min(),disk_mesh.Rout
        radii, angular_frequency_sq = disk.evaluate_angular_freq_centralgravity(R1,R2,Nvals=600)
        _, sound_speed = disk.evaluate_soundspeed(R1,R2,Nvals=600)
        pressure_midplane = dens0_profile(radii) * sound_speed**2
        _,pressure_midplane_gradient =  disk.evaluate_radial_gradient(pressure_midplane,R1,R2,Nvals=600)
        _,soundspeed_sq_gradient =  disk.evaluate_radial_gradient(sound_speed**2,R1,R2,Nvals=600)
        angular_frequency = np.sqrt(angular_frequency_sq + pressure_midplane_gradient/dens0_profile(radii)/radii)

        
        #interpolate mid-plane quantities
        vphi_profile = interp1d(radii,angular_frequency*radii,kind='linear')
        soundspeedsq_profile = interp1d(radii,sound_speed**2,kind='linear')
        soundspeedsq_gradient_profile = interp1d(radii,soundspeed_sq_gradient,kind='linear')

        #primitive variables
        vphi, press = np.zeros(R.shape),np.zeros(R.shape)
        ind = (R < disk_mesh.Rout) & (np.abs(z) < disk_mesh.zmax) 
        vphi[ind] = vphi_profile(R[ind]) -  soundspeedsq_gradient_profile(R[ind]) * np.log(dens[ind]/dens0_profile(R[ind]))
        press[ind] = dens[ind] * soundspeedsq_profile(R[ind])
        ind = (R >= disk_mesh.Rout) | (np.abs(z) > disk_mesh.zmax) 
        vphi[ind] = 0
        dens[ind] = dens_cut/10000
        press_cut = dens_cut * soundspeed(disk_mesh.Rout,disk.csnd0,disk.l,disk.csndR0)**2
        press[ind] = press_cut
        ind = R< disk_mesh.Rin 
        vphi[ind] = vphi[ind]*np.exp(-(disk_mesh.Rin-R[ind])**2/R[ind]**2)
        dens[ind] = dens_cut/100000
        ind = dens < dens_cut/100
        press[ind] = press_cut

        print "DENS_CUT",dens_cut
        print "PRESS_CUT",press_cut
        
        vr = np.zeros(R.shape)
        ids = np.arange(1,R.shape[0]+1,1)

        
        return R,phi,z,dens,vphi,vr,press,ids


    def incline(self,theta,phi,disk_mesh):
        costheta,sintheta = np.cos(theta*np.pi/180.0),np.sin(theta*np.pi/180.0)
        cosphi,sinphi = np.cos(phi*np.pi/180.0),np.sin(phi*np.pi/180.0)


        self.pos[:,0]-= 0.5 * disk_mesh.BoxSize
        self.pos[:,1]-= 0.5 * disk_mesh.BoxSize
        self.pos[:,2]-= 0.5 * disk_mesh.BoxSize
        
        R = np.sqrt(self.pos[:,0]**2+self.pos[:,1]**2+self.pos[:,2]**2)
        ind = R < 1.5 * disk_mesh.Rout 
        
        self.pos[ind,1],self.pos[ind,2] = costheta * (self.pos[ind,1]) - sintheta * self.pos[ind,2],\
                               sintheta * self.pos[ind,1] + costheta * self.pos[ind,2]
        self.pos[ind,0],self.pos[ind,1] = cosphi * self.pos[ind,0] - sinphi * self.pos[ind,1], \
                                sinphi * self.pos[ind,0] + cosphi * self.pos[ind,1]

        self.vel[ind,1],self.vel[ind,2] = costheta * self.vel[ind,1] - sintheta * self.vel[ind,2],\
                               sintheta * self.vel[ind,1] + costheta * self.vel[ind,2]
        self.vel[ind,0],self.vel[ind,1] = cosphi * self.vel[ind,0] - sinphi * self.vel[ind,1], \
                                sinphi * self.vel[ind,0] + cosphi * self.vel[ind,1]
        
        self.pos[:,0]+= 0.5 * disk_mesh.BoxSize
        self.pos[:,1]+= 0.5 * disk_mesh.BoxSize
        self.pos[:,2]+= 0.5 * disk_mesh.BoxSize

    def extract(self,index):
        self.pos=self.pos[index,:]
        self.vel=self.vel[index,:]
        self.dens=self.dens[index]
        self.utherm=self.utherm[index]
        self.ids=self.ids[index]

    def append(self,snapshot):
        self.pos=np.concatenate([self.pos,snapshot.pos],axis=0)
        self.vel=np.concatenate([self.vel,snapshot.vel],axis=0)
        self.dens=np.append(self.dens,snapshot.dens)
        self.utherm=np.append(self.utherm,snapshot.utherm)
        self.ids=np.append(self.ids,snapshot.ids)
        self.ids[self.ids > 0] = np.arange(1,1+self.ids[self.ids > 0].shape[0])


        
    def write_snapshot(self,disk,disk_mesh,filename="disk.dat.hdf5",time=0):
        
       
        
        Ngas = self.pos.shape[0]
        f=ws.openfile(filename)
        npart=np.array([Ngas,0,0,0,0,0], dtype="uint32")
        massarr=np.array([0,0,0,0,0,0], dtype="float64")
        header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr, time=time,
                              boxsize=disk_mesh.BoxSize, double = np.array([1], dtype="int32"))

        
        ws.writeheader(f, header)
        ws.write_block(f, "POS ", 0, self.pos)
        ws.write_block(f, "VEL ", 0, self.vel)
        ws.write_block(f, "MASS", 0, self.dens)
        ws.write_block(f, "U   ", 0, self.utherm)
        ws.write_block(f, "ID  ", 0, self.ids)
        ws.closefile(f)
        


if __name__=="__main__":


    d = disk()
    m = disk_mesh(d)
    m.create()
    ss = snapshot()
    ss.create(d,m)
    ss.write_snapshot(d,m)
    
