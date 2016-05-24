import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import snapHDF5 as ws
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import cumtrapz



def powerlaw_sigma(R,sigma0,p,R0):
    return sigma0*(R/R0)**(-p)

def similarity_sigma(R,sigma0,gamma,Rc):
    return sigma0*(R/Rc)**(-gamma) * np.exp(-(R/Rc)**(2.0-gamma))

def powerlaw_cavity_sigma(R,sigma0,p,xi,R_cav):
    return sigma0 * (R_cav/R)**p * np.exp(-(R_cav/R)**xi) 

def similarity_cavity_sigma(R,sigma0,gamma,Rc,xi,R_cav):
    return sigma0*(R/Rc)**(-gamma) * np.exp(-(R/Rc)**(2.0-gamma)) * np.exp(-(R_cav/R)**xi) 

def soundspeed(R,csnd0,l,R0):
    return csnd0 * (R/R0)**(-l*0.5)


def SplineProfile(R,h):
    r2 = R*R
    if(R >= h):
        wp = - 1.0 / R
    else:
        h_inv = 1.0 / h
        h3_inv = h_inv * h_inv * h_inv
        u = R * h_inv
        if(u < 0.5):
            wp =  h_inv * (-2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6)))
        else:
            wp = h_inv * (-3.2 + 0.066666666667 / u + u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u))))
      
    return -wp
  
def SplineDerivative(R,h):
    r2 = R * R
    fac = 0.0
    if(R >= h):
        fac = 1.0 / (r2 * R)
    else:
        h_inv = 1.0 / h
        h3_inv = h_inv * h_inv * h_inv
        u = R * h_inv
        if(u < 0.5):
            fac = h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4))
        else:
            fac = h3_inv * (21.333333333333 - 48.0 * u + 38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / \
                            (u * u * u))
            
    return fac

class powerlaw_disk(object):
    def __init__(self, *args, **kwargs):

        self.sigma0 = kwargs.get("sigma0")
        self.p = kwargs.get("p")
        self.R0 = kwargs.get("R0")


        #set default values
        if (self.sigma0 == None):
            self.sigma0 = 1.0
        if (self.p == None):
            self.p = 1.0
        if (self.R0 == None):
            self.R0 = 1.0

    def evaluate(self,R):
        return powerlaw_sigma(R,self.sigma0,self.p,self.R0)

class similarity_disk(object):
    def __init__(self, *args, **kwargs):
        self.sigma0 = kwargs.get("sigma0")
        self.gamma = kwargs.get("gamma")
        self.Rc = kwargs.get("Rc")

        #set default values
        if (self.sigma0 == None):
            self.sigma0 = 1.0
        if (self.gamma == None):
            self.gamma = 1.0
        if (self.Rc == None):
            self.Rc = 1.0

    def evaluate(self,R):
        return similarity_sigma(R,self.sigma0,self.gamma,self.Rc)

class powerlaw_cavity_disk(object):
    def __init__(self, *args, **kwargs):

        self.sigma0 = kwargs.get("sigma0")
        self.p = kwargs.get("p")
        self.R_cav = kwargs.get("R_cav")
        self.xi = kwargs.get("xi")


        #set default values
        if (self.sigma0 == None):
            self.sigma0 = 1.0
        if (self.p == None):
            self.p = 1.0
        if (self.R_cav == None):
            self.R_cav = 5.0
        if (self.xi == None):
            self.xi = 4.0

    def evaluate(self,R):
        return powerlaw_cavity_sigma(R,self.sigma0,self.p,self.R_cav,self.xi)


class similarity_cavity_disk(object):
    def __init__(self, *args, **kwargs):

        self.sigma0 = kwargs.get("sigma0")
        self.gamma = kwargs.get("gamma")
        self.Rc = kwargs.get("Rc")
        self.R_cav = kwargs.get("R_cav")
        self.xi = kwargs.get("xi")

        
        #set default values
        if (self.sigma0 == None):
            self.sigma0 = 1.0
        if (self.gamma == None):
            self.gamma = 1.0
        if (self.Rc == None):
            self.Rc = 1.0
        if (self.R_cav == None):
            self.R_cav = 5.0
        if (self.xi == None):
            self.xi = 4.0

    def evaluate(self,R):
        return similarity_cavity_sigma(R,self.sigma0,self.gamma,self.Rc,self.R_cav,self.xi)


class disk(object):
    def __init__(self, *args, **kwargs):
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
        self.quadrupole_correction =  kwargs.get("quadrupole_correction")

        

        #set defaults 
        if (self.sigma_type == None):
            self.sigma_type="powerlaw"
        if (self.l == None):
            self.l = 1.0
        if (self.csnd0 == None):
            self.csnd0 = 0.05
        if (self.adiabatic_gamma == None):
            self.adiabatic_gamma = 7.0/5
        if (self.effective_gamma == None):
            self.effective_gamma = 1.0
        if (self.alphacoeff == None):
            self.alphacoeff = 0.01
        if (self.Mcentral == None):
            self.Mcentral = 1.0
        if (self.quadrupole_correction == None):
            self.quadrupole_correction = 0
            
        if (self.sigma_type == "powerlaw"):
            self.sigma_disk = powerlaw_disk(**kwargs)
            if (self.csndR0 == None):
                self.csndR0 = self.sigma_disk.R0

        if (self.sigma_type == "similarity"):
            self.sigma_disk = similarity_disk(**kwargs)
            if (self.csndR0 == None):
                self.csndR0 = self.sigma_disk.Rc

        if (self.sigma_type == "powerlaw_cavity"):
            self.sigma_disk = powerlaw_cavity_disk(**kwargs)
            if (self.csndR0 == None):
                self.csndR0 = self.sigma_disk.R_cav

        if (self.sigma_type == "similarity_cavity"):
            self.sigma_disk = similarity_cavity_disk(**kwargs)
            if (self.csndR0 == None):
                self.csndR0 = self.sigma_disk.Rc

                
        if (self.sigma_cut == None):
            self.sigma_cut = self.sigma_disk.sigma0 * 1e-7
        
            
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
    
    def evaluate_vertical_structure(self,R,zin,zout,Nzvals=300):
        def integrand(z): return np.exp(-self.vertical_potential(R,z)/soundspeed(R,self.csnd0,self.l,self.csndR0)**2)
        VertProfileNorm =  self.sigma_disk.evaluate(R)/(2.0*quad(integrand,0,zout*100)[0])
        zvals = np.logspace(np.log10(zin),np.log10(zout),Nzvals)
        zrho = [VertProfileNorm * np.exp(-self.vertical_potential(R,zz)/soundspeed(R,self.csnd0,self.l,self.csndR0)**2) for zz in zvals]
        return zvals,zrho,VertProfileNorm

    def evaluate_enclosed_vertical(self,R,zin,zout,Nzvals=300):
        zvals, zrho,_ = self.evaluate_vertical_structure(R,zin,zout,Nzvals)
        zmass = np.append(0,cumtrapz(zrho,zvals))
        return zvals, zmass
    
    def spherical_potential(self,r,soft=0.01):
        #print "hehe",SplineProfile(r,soft)
        return SplineProfile(r,soft)
    
    def vertical_potential(self,R,z,M_central=1.0,G=1,soft=0.01):
        return -G*M_central * (self.spherical_potential(np.sqrt(R*R + z*z),soft) - self.spherical_potential(R,soft))

    def solve_vertical_structure(self,R,z,Rin,Rout,Ncells):

        dens = np.zeros(R.shape[0])
        
        if (Ncells < 50000): R_bins = 120
        elif (Ncells < 100000): R_bins = 160
        elif (Ncells < 200000): R_bins = 250
        elif (Ncells < 600000): R_bins   = 500
        else: R_bins = 700
        radial_bins = self.evaluate_radial_mass_bins(Rin,Rout,R_bins)
        #fix the bins a bit
        print radial_bins
        dRin = radial_bins[1]-radial_bins[0]
        radial_bins = np.append(np.arange(0,radial_bins[1],dRin/10),radial_bins[1:])
        print radial_bins
        
        bin_inds=np.digitize(R,radial_bins)
        mid_plane = []
        radii = []
        zin,zout = 0.99*np.abs(z).min(),1.01*np.abs(z).max()
        for kk in range(0,radial_bins.shape[0]):
            N_in_bin = R[bin_inds == kk].shape[0]
            print kk,R_bins,radial_bins[kk]
            if (N_in_bin == 0):
                mid_plane.append(0.0)
                radii.append(radial_bins[kk])
                continue
            bin_radius = R[bin_inds == kk].mean()
            zvals,zrhovals,rho0 = self.evaluate_vertical_structure(bin_radius,zin,zout,Nzvals=500)
            mid_plane.append(rho0)
            radii.append(bin_radius)
            dens_profile = interp1d(zvals,zrhovals,kind='linear')
            print kk,R_bins,bin_radius,zvals.min(),zvals.max(),np.abs(z[bin_inds == kk]).min(),np.abs(z[bin_inds == kk]).max()
            dens[bin_inds == kk] = dens_profile(np.abs(z[bin_inds == kk]))

        return dens,np.array(radii),np.array(mid_plane)
        
    
class disk_mesh():
    def __init__(self, *args, **kwargs):

        self.mesh_type=kwargs.get("mesh_type")
        self.Rin = kwargs.get("Rin")
        self.Rout = kwargs.get("Rout")
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
        if (self.mesh_type == None):
            self.mesh_type="polar"
        if (self.Rin == None):
            self.Rin = 1
        if (self.Rout == None):
            self.Rout = 10
        if (self.NR == None):
            self.NR = 800
        if (self.Nphi == None):
            self.Nphi = 600
        if (self.Ncells == None):
            self.Ncells =  self.NR * self.Nphi
        if (self.N_inner_boundary_rings == None):
            self.N_inner_boundary_rings = 1
        if (self.N_outer_boundary_rings == None):
            self.N_outer_boundary_rings = 1            
            
        if (self.BoxSize == None):
            self.BoxSize = 1.2 * 2* self.Rout
            
        if (self.fill_box == None):
            self.fill_box = False
        if (self.fill_center == None):
            self.fill_center = False   
        if (self.fill_background == None):
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

                R = np.append(R,Rcenter)
                phi = np.append(phi,phicenter)

            z = np.zeros(R.shape[0])
                
            return R,phi,z

        if (self.mesh_type == "mc"):
            R,phi = self.mc_sample_2d(disk)
            z = self.mc_sample_vertical(R,disk)


            if (self.fill_background == True):
                Rback,phiback = self.mc_sample_2d(disk,Npoints=0.15 * self.Ncells)
                zback = self.mc_sample_vertical_background(R,Rback,z,disk)
                Rbackmax = Rback.max()
                R = np.append(R,Rback)
                phi = np.append(phi,phiback)
                z = np.append(z,zback)
                
                Lx,Ly,Lz = 2*Rbackmax,2*Rbackmax,2*zback.max()
                delta = zback.max()/4
                xback,yback,zback = self.sample_fill_box(Lx,Ly,Lz,delta)
                Rback = np.sqrt(xback**2+yback**2)
                phiback = np.arctan2(yback,xback)
                ind = Rback > Rbackmax
                Rback, phiback,zback = Rback[ind], phiback[ind],zback[ind]

                R = np.append(R,Rback)
                phi = np.append(phi,phiback)
                z = np.append(z,zback)

                
            if (self.fill_center == True):
                Rmin  = R.min()
                Rmax  = R.max()
                zmax  = z.max()
                rvals,mvals = disk.evaluate_enclosed_mass(self.Rin, self.Rout,Nvals=100)
                cellmass = mvals[-1]/self.Ncells
                m2r=interp1d(np.append([0],mvals),np.append([0],rvals),kind='linear')
                Rmin = m2r(cellmass)
                sigma_in = disk.sigma_disk.evaluate(Rmin)
                if (sigma_in < disk.sigma_cut): sigma_in = disk.sigma_cut
                rho_in = sigma_in/zmax
                delta = 0.3*(cellmass/rho_in)**0.333333
                Lx, Ly, Lz = 2 * Rmin, 2 * Rmin,2*zmax
                xcenter,ycenter,zcenter =  self.sample_fill_box(Lx,Ly,Lz,delta)
                Rcenter = np.sqrt(xcenter**2+ycenter**2)
                phicenter = np.arctan2(ycenter,xcenter)
                ind = Rcenter < Rmin 
                Rcenter, phicenter,zcenter = Rcenter[ind], phicenter[ind],zcenter[ind]

                R = np.append(R,Rcenter)
                phi = np.append(phi,phicenter)
                z = np.append(z,zcenter)

            if (self.fill_box == True):
                print "hello"
                Rmax = R.max()
                zmax = np.abs(z).max()
                Nlayers = 1
                if (self.BoxSize > 3 * Rmax):
                    Nlayers+=1
                    if  (self.BoxSize > 9 * Rmax):
                        Nlayers+1
                for jj in range(Nlayers):
                    Lx, Ly, Lz = 3 * Rmax, 3 * Rmax, 3 * Rmax
                    delta  = 0.3 * Rmax
                    xcenter,ycenter,zcenter =  self.sample_fill_box(Lx,Ly,Lz,delta)
                    ind = (np.abs(xcenter) > Rmax) & (np.abs(xcenter) < 0.5 * self.BoxSize) &\
                          (np.abs(ycenter) > Rmax) & (np.abs(ycenter) < 0.5 * self.BoxSize) &\
                          (np.abs(zcenter) > zmax) & (np.abs(zcenter) < 0.5 * self.BoxSize)
                    R = np.append(R,np.sqrt(xcenter**2+ycenter**2))
                    phi = np.append(phi,np.arctan2(ycenter,xcenter))
                    z=np.append(z,zcenter)
                    Rmax,zmax = 3 * Rmax, 3*Rmax
                    
            
 
            return R,phi,z

                             
    def mc_sample_2d(self,disk,**kwargs):

        Npoints = kwargs.get("Npoints")
        if (Npoints == None): Npoints = self.Ncells
        
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

                
    def sample_fill_box(self,Lx,Ly,Lz,delta):
        
        xbox,ybox,zbox = np.meshgrid(np.arange(-0.49 * Lx+0.5*delta, 0.49 * Lx,delta),
                                     np.arange(-0.49 * Ly+0.5*delta, 0.49 * Ly,delta),
                                     np.arange(-0.49 * Lz+0.5*delta, 0.49 * Lz,delta))
        
        xbox,ybox,zbox =  xbox.flatten(),ybox.flatten(),zbox.flatten()
        
        return xbox,ybox,zbox

class snapshot():

    def __init__(self,*args,**kwargs):
        self.pos=kwargs.get("pos")
        self.vel=kwargs.get("vel")
        self.dens=kwargs.get("dens")
        self.utherm=kwargs.get("utherm")
        self.ids=kwargs.get("ids")

    def create(self,disk,disk_mesh):
        
        R,phi,z,dens,vphi,vr,press,ids = self.assign_primitive_variables(disk,disk_mesh)
        
        self.load(R,phi,z,dens,vphi,vr,press,ids,disk_mesh.BoxSize,disk.adiabatic_gamma)

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
        self.utherm = press/self.dens/(adiabatic_gamma - 1)
        self.ids = ids
    
    
     
    def assign_primitive_variables(self,disk,disk_mesh):
        
        R,phi,z = disk_mesh.create(disk)

        R1,R2 = 1e-4,1.5*disk_mesh.Rout
        #obtain density of cells
        dens, radii, midplane_dens = disk.solve_vertical_structure(R,z,R1,R2,disk_mesh.Ncells)
        dens_cut = midplane_dens[-1]
        dens[dens < dens_cut] = dens_cut
        midplane_dens[midplane_dens < dens_cut] = dens_cut
        dens0_profile =  interp1d(radii,midplane_dens,kind='linear')

        #evaluate other quantities
        R1,R2 = 0.99*R.min(),disk_mesh.Rout
        radii, angular_frequency_sq = disk.evaluate_angular_freq_centralgravity(R1,R2)
        _, sound_speed = disk.evaluate_soundspeed(R1,R2)
        print radii.min(),radii.max()
        pressure_midplane = dens0_profile(radii) * sound_speed**2
        _,pressure_midplane_gradient =  disk.evaluate_radial_gradient(pressure_midplane,R1,R2)
        _,soundspeed_sq_gradient =  disk.evaluate_radial_gradient(sound_speed**2,R1,R2)
        print radii.shape, sound_speed.shape,pressure_midplane.shape,pressure_midplane_gradient.shape,soundspeed_sq_gradient.shape
        angular_frequency = np.sqrt(angular_frequency_sq + pressure_midplane_gradient/dens0_profile(radii)/radii)


        #interpolate mid-plane quantities
        vphi_profile = interp1d(radii,angular_frequency*radii,kind='linear')
        soundspeedsq_profile = interp1d(radii,sound_speed,kind='linear')
        soundspeedsq_gradient_profile = interp1d(radii,soundspeed_sq_gradient,kind='linear')

        #primitive variables
        vphi, press = np.zeros(R.shape),np.zeros(R.shape)
        ind = R < disk_mesh.Rout
        vphi[ind] = vphi_profile(R[ind]) -  soundspeedsq_gradient_profile(R[ind]) * np.log(dens[ind]/dens0_profile(R[ind]))
        press[ind] = dens[ind] * soundspeedsq_profile(R[ind])
        ind = R >= disk_mesh.Rout
        vphi[ind] = 0
        dens[ind] = dens_cut
        press[ind] = dens_cut * soundspeed(disk_mesh.Rout,disk.csnd0,disk.l,disk.csndR0)**2
        
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
        print 2 * disk_mesh.Rout,disk_mesh.BoxSize 
        print R
        
        print "hello",self.pos[ind,:].shape
        
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
    
