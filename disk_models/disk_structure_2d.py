import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import snapHDF5 as ws
from scipy.interpolate import interp1d

from disk_density_profiles import *
from disk_external_potentials import *
from disk_other_functions import *
from disk_snapshot import *

class disk2d(object):
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

        #corrections to density profile by a gap
        self.add_gap = kwargs.get("add_gap")
        if (self.add_gap is True):
            self.gap_center = kwargs.get("gap_center")
            self.gap_width = kwargs.get("gap_width")
            self.gap_depth = kwargs.get("gap_depth")
            self.gap_steep = kwargs.get("gap_steep")



            
        #set defaults ###############################
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
        if (self.quadrupole_correction is None):
            self.quadrupole_correction = 0
            
        if (self.sigma_type == "powerlaw"):
            self.sigma_disk = powerlaw_disk(**kwargs)
            if (self.csndR0 is None):
                self.csndR0 = self.sigma_disk.R0

        if (self.sigma_type == "powerlaw_zerotorque"):
            self.sigma_disk = powerlaw_zerotorque_disk(**kwargs)
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

        if (self.sigma_cut is None):
            self.sigma_cut = self.sigma_disk.sigma0 * 1e-7

        if (self.add_gap is None):
            self.add_gap = False
        if (self.add_gap is True):
            if (self.gap_center is None):
                self.gap_center = 1.0
            if (self.gap_width is None):
                self.gap_width = 0.1
            if (self.gap_depth is None):
                self.gap_depth = 0.01
            if (self.gap_steep is None):
                self.gap_steep = 4

            
    def evaluate_sigma(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        sigma = self.sigma_disk.evaluate(rvals)

        if (self.add_gap):
            print self.gap_center,self.gap_width,self.gap_depth,self.gap_steep
            sigma = sigma * gap_profile(rvals,self.gap_center,self.gap_width,self.gap_depth,self.gap_steep)
            
        sigma[sigma < self.sigma_cut] = self.sigma_cut
        return rvals,sigma
    
    def evaluate_soundspeed(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        return rvals,soundspeed(rvals,self.csnd0,self.l,self.csndR0)

    def evaluate_pressure(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals,sigma = self.evaluate_sigma(Rin,Rout,Nvals,scale=scale)
        return rvals, sigma**(self.effective_gamma) * \
            self.evaluate_soundspeed(Rin,Rout,Nvals,scale=scale)[1]**2

    def evaluate_viscosity(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals,csnd =  self.evaluate_soundspeed(Rin,Rout,Nvals,scale=scale)
        Omega_sq = self.Mcentral/rvals**3 * (1 + 3 * self.quadrupole_correction/rvals**2)
        nu = self.alphacoeff * csnd * csnd / np.sqrt(Omega_sq)
        return rvals, nu
    
    def evaluate_pressure_gradient(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals, press = self.evaluate_pressure(Rin,Rout,Nvals,scale=scale)
        _, dPdR = self.evaluate_radial_gradient(press,Rin,Rout,Nvals,scale=scale)
        return rvals,dPdR

    def evaluate_rotation_curve(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals = self.evaluate_radial_zones(Rin,Rout,Nvals,scale)
        Omega_sq = self.Mcentral/rvals**3 * (1 + 3 * self.quadrupole_correction/rvals**2)
        
        return rvals, np.sqrt(Omega_sq + self.evaluate_pressure_gradient(Rin,Rout,Nvals,scale=scale)[1] / \
            self.evaluate_sigma(Rin,Rout,Nvals,scale=scale)[1]/ rvals)

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

            
class disk_mesh2d(disk):
    def __init__(self, *args, **kwargs):

        self.mesh_type=kwargs.get("mesh_type")
        self.Rin = kwargs.get("Rin")
        self.Rout = kwargs.get("Rout")
        self.NR = kwargs.get("NR")
        self.Nphi = kwargs.get("Nphi")
        self.BoxSize = kwargs.get("BoxSize")
        self.mesh_alignment = kwargs.get("mesh_alignment")
        self.N_inner_boundary_rings = kwargs.get("N_inner_boundary_rings")
        self.N_outer_boundary_rings = kwargs.get("N_outer_boundary_rings") 
        self.fill_box = kwargs.get("fill_box")
        self.fill_center = kwargs.get("fill_center")
        self.fill_box_Nmax = kwargs.get("fill_box_Nmax")
        
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
        if (self.fill_box_Nmax is None):
            self.fill_box_Nmax = 64

    def create(self,*args,**kwargs):

        
        if (self.mesh_type == "polar"):

            rvals = np.logspace(np.log10(self.Rin),np.log10(self.Rout),self.NR+1)
            rvals = rvals[:-1] + 0.5 * np.diff(rvals)
            self.deltaRin,self.deltaRout = rvals[1]-rvals[0],rvals[-1]-rvals[-2]
            # Add cells outside the boundary
            for kk in range(self.N_inner_boundary_rings): rvals=np.append(rvals[0]-self.deltaRin, rvals)
            for kk in range(self.N_outer_boundary_rings): rvals=np.append(rvals,rvals[-1]+self.deltaRout)

            phivals = np.linspace(0,2*np.pi,self.Nphi+1)
            R,phi = np.meshgrid(rvals,phivals)
            
            if (self.mesh_alignment == "interleaved"):
                phi[:-1,4*self.N_inner_boundary_rings:-2*self.N_outer_boundary_rings:2] = phi[:-1,4*self.N_inner_boundary_rings:-2*self.N_outer_boundary_rings:2] + 0.5*np.diff(phi[:,4*self.N_inner_boundary_rings:-2*self.N_outer_boundary_rings:2],axis=0)
                
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
                if (self.BoxSize/interval > self.fill_box_Nmax):
                    interval = self.BoxSize/self.fill_box_Nmax
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

                
            return R,phi


class snapshot():

    def __init__(self,*args,**kwargs):
        self.pos=kwargs.get("pos")
        self.vel=kwargs.get("vel")
        self.dens=kwargs.get("dens")
        self.utherm=kwargs.get("utherm")
        self.ids=kwargs.get("ids")

    def create(self,disk,disk_mesh):
        
        R,phi,dens,vphi,vr,press,ids = self.assign_primitive_variables(disk,disk_mesh)
        
        self.load(R,phi,dens,vphi,vr,press,ids,disk_mesh.BoxSize,disk.adiabatic_gamma)

    def load(self,R,phi,dens,vphi,vr,press,ids,BoxSize,adiabatic_gamma):
        
        x = R * np.cos(phi) + 0.5 * BoxSize
        y = R * np.sin(phi) + 0.5 * BoxSize
        z = np.zeros(x.shape[0])
        
        vx = vr * np.cos(phi) - vphi * np.sin(phi)
        vy = vr * np.sin(phi) + vphi * np.cos(phi)
        vz = np.zeros(vx.shape[0])
        
        self.dens = dens
        self.pos = np.array([x,y,z]).T
        self.vel = np.array([vx,vy,vz]).T
        self.utherm = press/self.dens/(adiabatic_gamma - 1)
        self.ids = ids
    
    
     
    def assign_primitive_variables(self,disk,disk_mesh):
        
        R,phi = disk_mesh.create()
        
        R1,R2 = 0.99*R.min(),1.01*R.max()
        radii, density = disk.evaluate_sigma(R1,R2)
        _, angular_frequency = disk.evaluate_rotation_curve(R1,R2)
        _, radial_velocity = disk.evaluate_radial_velocity(R1,R2)
        _, pressure = disk.evaluate_pressure(R1,R2)
        
        dens_profile = interp1d(radii,density,kind='linear')
        vphi_profile = interp1d(radii,angular_frequency*radii,kind='linear')
        vr_profile = interp1d(radii,radial_velocity,kind='linear')
        press_profile = interp1d(radii,pressure,kind='linear')

        #primitive variables
        dens = dens_profile(R)
        vphi = vphi_profile(R)
        vr = vr_profile(R)
        press = press_profile(R)

        #cell ids
        ids = np.arange(1,R.shape[0]+1,1)
        #check for boundaries, first inside the boundary
        ind_inner = (R > disk_mesh.Rin) & (R < (disk_mesh.Rin + disk_mesh.N_inner_boundary_rings * disk_mesh.deltaRin))
        ind_outer = (R < disk_mesh.Rout) & (R > (disk_mesh.Rout - disk_mesh.N_outer_boundary_rings * disk_mesh.deltaRout))
        ids[ind_inner+ind_outer] = -1
        #now outside the boundary
        ind_inner = (R < disk_mesh.Rin) & (R > (disk_mesh.Rin - disk_mesh.N_inner_boundary_rings * disk_mesh.deltaRin))
        ind_outer = (R > disk_mesh.Rout) & (R < (disk_mesh.Rout + disk_mesh.N_outer_boundary_rings * disk_mesh.deltaRout))
        ids[ind_inner+ind_outer] = -2
        #or buffer cells (only if there are boundaries at the interface)
        ind_inner = (R < (disk_mesh.Rin - disk_mesh.N_inner_boundary_rings * disk_mesh.deltaRin))
        ind_outer = (R > (disk_mesh.Rout + disk_mesh.N_outer_boundary_rings * disk_mesh.deltaRout))
        if (disk_mesh.N_inner_boundary_rings > 0):  ids[ind_inner] = range(-3,-3-len(ids[ind_inner]),-1)
        if (disk_mesh.N_outer_boundary_rings > 0):  ids[ind_outer] = range(-3,-3-len(ids[ind_outer]),-1)

        dens[ind_inner+ind_outer] = disk.sigma_cut
        vr[dens <= disk.sigma_cut] = 0
        vphi[dens <= disk.sigma_cut] = 0
        press[dens <= disk.sigma_cut] = press[dens <= disk.sigma_cut].min()


        vphi[ids < -2] = 0
        vr[ids < -2] = 0
        
        return R,phi,dens,vphi,vr,press,ids


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
        

    def summary(self,disk,disk_mesh):

        print("Created snapshot with NR=%i, Nphi=%i and a total of Ncells=%i" \
              % (disk_mesh.NR,disk_mesh.Nphi,self.pos.shape[0]))
        print("SUMMARY:")
        print("\t Rin=%f Rout=%f" %( disk_mesh.Rin, disk_mesh.Rout))

if __name__=="__main__":


    d = disk()
    m = disk_mesh(d)
    m.create()
    ss = snapshot()
    ss.create(d,m)
    ss.write_snapshot(d,m)
    
