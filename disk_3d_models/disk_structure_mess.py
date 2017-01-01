import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import snapHDF5 as ws
from scipy.interpolate import interp1d
from scipy.integrate import quad

def powerlaw_sigma(R,sigma0,p,R0):
    return sigma0*(R/R0)**(-p)

def similarity_sigma(R,sigma0,gamma,Rc):
    return sigma0*(R/Rc)**(-gamma) * np.exp(-(R/Rc)**(2.0-gamma))

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

class disk(object):
    def __init__(self, *args, **kwargs):
        #define the properties of the axi-symmetric disk model
        self.sigma_type = kwargs.get("sigma_type")

        #Temperature profile properties
        self.csndR0 = kwargs.get("csndR0") #reference radius
        self.csnd0 = kwargs.get("csnd0") # soundspeed scaling
        self.l = kwargs.get("l") # temperature profile index

        #thermodynamic parameters
        self.adiabatic_gamma = kwargs.get("adiabatic_gamma")
        self.effective_gamma = kwargs.get("effective_gamma")        
        
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
        if (self.Mcentral == None):
            self.Mcentral = 1.0
        if (self.quadrupole_correction == None):
            self.quadrupole_correction = 0
            
        if (self.sigma_type == "powerlaw"):
            print kwargs
            self.sigma_disk = powerlaw_disk(**kwargs)
            if (self.csndR0 == None):
                self.csndR0 = self.sigma_disk.R0

        if (self.sigma_type == "similarity"):
            self.sigma_disk = similarity_disk(**kwargs)
            if (self.csndR0 == None):
                self.csndR0 = self.sigma_disk.Rc

            
    def evaluate_sigma(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", scale, "not known!"
            sys.exit() 
        return rvals,self.sigma_disk.evaluate(rvals)
    
    def evaluate_soundspeed(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", scale, "not known!"
            sys.exit()
        return rvals,soundspeed(rvals,self.csnd0,self.l,self.csndR0)

    def evaluate_pressure(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", scale, "not known!"
            sys.exit()
        return rvals, self.evaluate_sigma(Rin,Rout,Nvals,scale=scale)[1]**(self.effective_gamma) * \
            self.evaluate_soundspeed(Rin,Rout,Nvals,scale=scale)[1]**2

    def evaluate_viscosity(self,Rin,Rout,Nvals=1000,scale='log'):

        rvals,csnd =  self.evaluate_soundspeed(Rin,Rout,Nvals,scale=scale)
        Omega_sq = self.Mcentral/rvals**3 * (1 + 3 * self.quadrupole_correction/rvals**2)
        nu = self.alphacoeff * csnd * csnd / np.sqrt(Omega_sq)
        
        return rvals, nu
        
    
    def evaluate_pressure_gradient(self,Rin,Rout,Nvals=1000,scale='log'):
        rvals, press = self.evaluate_pressure(Rin,Rout,Nvals,scale=scale)
        if (scale == 'log'):
            dPdlogR = np.gradient(press)/np.gradient(np.log10(rvals))
            dPdR = dPdlogR/rvals/np.log(10)
        elif (scale == 'linear'):
            dPdR = np.gradient(press)/np.gradient(rvals)
        return rvals,dPdR

    def evaluate_rotation_curve(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", scale, "not known!"
            sys.exit()
        
        Omega_sq = self.Mcentral/rvals**3 * (1 + 3 * self.quadrupole_correction/rvals**2)

        return rvals, Omega_sq + self.evaluate_pressure_gradient(Rin,Rout,Nvals,scale=scale)[1] / \
            self.evaluate_sigma(Rin,Rout,Nvals,scale=scale)[1]/ rvals

    def evaluate_radial_velocity(self,Rin,Rout,Nvals=1000,scale='log'):
        if (scale == 'log'):
            rvals = np.logspace(np.log10(Rin),np.log10(Rout),Nvals)
        elif (scale == 'linear'):
            rvals = np.linspace(Rin,Rout,Nvals)
        else: 
            print "[error] scale type ", scale, "not known!"
            sys.exit()
            
        return rvals, np.zeros(rvals.shape[0])

    def evaluate_radial_velocity_viscous(self,Rin,Rout,Nvals=1000,scale='log'):

        dOmegadR=np.gradient(Omega(R),dR,edge_order=2)
        dOmegadR=gradient_second_order(Omega(R),dR)
                    
        print dOmegadR
        velr=gradient_second_order(nu(R)*Sigma(R)*R**3*dOmegadR,dR)/R/Sigma(R)/gradient_second_order(R**2*Omega(R),dR)

        
        return rvals,velr

    def evaluate_radial_velocity_constant_mdot(self,Rin,Rout,Nvals=1000,scale='log'):
        
        return 0
        

    def evaluate_vertical_structure(self,R,zin,zout,Nzvals=200):
        def integrand(z): return np.exp(-vertical_potential(R,z)/soundspeed(R,self.csnd0,self.l,self.csndR0)**2)
        VertProfileNorm = vecSigma(radius)/(2.0*quad(integrand,0.0,zout*1000)[0])
        return 0
    
    def spherical_potential(self,r,soft=0.01):
        return SplineProfile(np.sqrt(R*R + z*z),soft
                             
    def vertical_potential(self,R,z,M_central=1.0,G=1,soft=0.01):
        return -G*M_central * (self.spherical_potential(np.sqrt(R*R + z*z),soft) - self.spherical_potential(R,soft))

                             
        
            
class disk_mesh(disk):
    def __init__(self, *args, **kwargs):

        self.mesh_type=kwargs.get("mesh_type")
        self.Rin = kwargs.get("rin")
        self.Rout = kwargs.get("rout")
        self.NR = kwargs.get("NR")
        self.Nphi = kwargs.get("Nphi")
        self.BoxSize = kwargs.get("BoxSize")
        self.mesh_alignment = kwargs.get("mesh_alignment")
        self.N_inner_boundary_rings = kwargs.get("N_inner_boundary_rings")
        self.N_outer_boundary_rings = kwargs.get("N_outer_boundary_rings") 
        self.fill_box = kwargs.get("fill_box")
        
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
        if (self.N_inner_boundary_rings == None):
            self.N_inner_boundary_rings = 1
        if (self.N_outer_boundary_rings == None):
            self.N_outer_boundary_rings = 1            
            
        if (self.BoxSize == None):
            self.BoxSize = 1.2 * 2* self.Rout
            
        if (self.fill_box == None):
            self.fill_box = False
            
    def create(self,*args,**kwargs):

        
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
                print "hello"
                #exponentially decreasing mesh density
                Rbackmin, Rbackmax, Rscale = R.max()+self.deltaRout, np.sqrt(2)/2*self.BoxSize, 0.2* self.BoxSize
                Nbackcells = np.ceil(Rscale * self.Nphi/self.deltaRout * (1 - np.exp((Rbackmin-Rbackmax)/Rscale)))
                #invert cummulative function
                rvals = Rbackmin - Rscale*np.log(1 - np.arange(Nbackcells)/Nbackcells)
                phivals = rd.random(Nbackcells)*2*np.pi
                phivals[rvals < 0.4*(Rbackmax+Rbackmin)] = np.round(phivals[rvals < 0.4*(Rbackmax+Rbackmin)],decimals=2)

                ind_inbox = (np.abs(rvals*np.cos(phivals)) < 0.5 * self.BoxSize) & (np.abs(rvals*np.sin(phivals)) < 0.5 * self.BoxSize)
                rvals, phivals = rvals[ind_inbox], phivals[ind_inbox]

                R = np.append(R,rvals)
                phi = np.append(phi,phivals)
                
            return R,phi
                             
        if (self.mesh_type = "mc"):
                             
            R,phi = mc_sample_2d()

            return R,phi

                             
    def mc_sample_2d(self,*args,**kwargs):

        # Montecarlo mesh parameters
        if (N_cells < 50000): R_bins = 100
        elif (N_cells < 100000): R_bins = 200
        elif (N_cells < 200000): R_bins = 400
        elif (N_cells < 600000): R_bins   = 600
        
        
        
        print "\nIntegrating vertical structure..."
        print "   Looping over ",R_bins," radial bins..."

        
        for k in range(0,INTERPOL_BINS):
        
    def fill_background(self,Ncells,Rmin,Rmax,zmin,zmax):
        

        
        

        
class snapshot():

    def __init__(self,*args,**kwargs):
        self.pos=kwargs.get("pos")
        self.vel=kwargs.get("vel")
        self.dens=kwargs.get("dens")
        self.utherm=kwargs.get("utherm")
        self.ids=kwargs.get("ids")

    def create(self,disk,disk_mesh):
        
        R,phi,self.dens,vphi,vr,press,self.ids = self.assign_primitive_variables(disk,disk_mesh)
        self.utherm = press/self.dens/(disk.adiabatic_gamma - 1)
        
        x = R * np.cos(phi) + 0.5 * disk_mesh.BoxSize
        y = R * np.sin(phi) + 0.5 * disk_mesh.BoxSize
        z = np.zeros(x.shape[0])
        
        vx = vr * np.cos(phi) - vphi * np.sin(phi)
        vy = vr * np.sin(phi) + vphi * np.cos(phi)
        vz = np.zeros(vx.shape[0])
        
        self.pos = np.array([x,y,z]).T
        self.vel = np.array([vx,vy,vz]).T

        
    def assign_primitive_variables(self,disk,disk_mesh):
        
        
        R,phi = disk_mesh.create()
        
        radii, density = disk.evaluate_sigma(R.min(),R.max())
        _, angular_frequency = disk.evaluate_rotation_curve(R.min(),R.max())
        _, radial_velocity = disk.evaluate_radial_velocity(R.min(),R.max())
        _, pressure = disk.evaluate_pressure(R.min(),R.max())
        
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
        #or buffer cells
        ind_inner = (R < (disk_mesh.Rin - disk_mesh.N_inner_boundary_rings * disk_mesh.deltaRin))
        ind_outer = (R > (disk_mesh.Rout + disk_mesh.N_outer_boundary_rings * disk_mesh.deltaRout))
        ids[ind_inner+ind_outer] = -3

        #print ids[ids < 0]
        
        return R,phi,dens,vphi,vr,press,ids


    def write_snapshot(self,disk,disk_mesh,filename="disk.dat.hdf5"):
        
       
        
        Ngas = self.pos.shape[0]
        f=ws.openfile(filename)
        npart=np.array([Ngas,0,0,0,0,0], dtype="uint32")
        massarr=np.array([0,0,0,0,0,0], dtype="float64")
        header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr,
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
    
