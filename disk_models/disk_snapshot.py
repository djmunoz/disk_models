import numpy as np
import matplotlib.pyplot as plt
from disk_hdf5 import snapHDF5 as ws
import itertools, sys

from scipy.integrate import quad
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
try:
    from scipy.spatial import Voronoi
except ImportError:
    None
    
import scipy.integrate as integ

from disk_parameter_files import *
from disk_particles import *


STAR_PARTTYPE = 4

class gas_data():
    def __init__(self,*args,**kwargs):
        self.pos=kwargs.get("pos")
        self.vel=kwargs.get("vel")
        self.dens=kwargs.get("dens")
        self.mass=kwargs.get("mass")
        self.press=kwargs.get("press")
        self.utherm=kwargs.get("utherm")
        self.ids=kwargs.get("ids")

    


class snapshot():

    def __init__(self,*args,**kwargs):

        self.gas = gas_data()
        self.particle = particle_data()
        
        self.params = paramfile(init_cond_file="./disk.dat")

        self.BoxSize = kwargs.get("BoxSize")

        
    def create(self,disk,disk_mesh,empty=False):

        self.BoxSize = disk_mesh.BoxSize
        
        # Obtain the primitive quantities for all cells
        if (empty == False):
            R,phi,z,dens,vphi,vr,press,ids = self.assign_primitive_variables(disk,disk_mesh)
        else:
            # Just a set of locations
            R, phi, z = disk_mesh.create(disk)
            dens,vphi,vr,press = None, None, None, None
            ids = np.arange(1, R.shape[0] + 1,1)

            
        # Load them into the snapshot
        if (disk.__class__.__name__ == 'disk3d'):
            dims = 3
        if (disk.__class__.__name__ == 'disk2d'):
            dims = 2
        self.load(R,phi,z,dens,None,vphi,vr,press,ids,dims=dims,adiabatic_gamma=disk.adiabatic_gamma)

        
        # Check if there is a central particle
        if (disk.central_particle):
            central_particle = particle_data()
            central_particle.add_particle(x = (0.5 * self.BoxSize),
                                          y = (0.5 * self.BoxSize),
                                          z = (0.5 * self.BoxSize),
                                          vx=0,
                                          vy=0,
                                          vz=0,
                                          m = disk.Mcentral, ID=(self.gas.ids.max()+1))
            #central_particle = particle_data(pos=np.array([0.5 * self.BoxSize, 0.5 * self.BoxSize, 0.5 * self.BoxSize]).reshape(1,3),
            #                                 vel=np.array([0.0,0.0,0.0]).reshape(1,3),
            #                                 mass = disk.Mcentral,ids=[np.array([self.gas.ids.max()+1]).T])
            self.load_particles(central_particle)


        if (empty == False):
            self.obtain_parameters(disk, disk_mesh, R, phi, z,dens,press)

    def obtain_parameters(self, disk, disk_mesh,R,phi,z,dens,press):
    
        # Obtain target masses and allowed volumes
        self.params.reference_gas_part_mass = disk.compute_disk_mass(disk_mesh.Rin,disk_mesh.Rout)/disk_mesh.Ncells

        ff = 1.0
        ind = (R < ff * 1.2 * disk_mesh.Rout) & ((R > disk_mesh.Rout))
        while (R[ind].shape[0] < 10):
            ind = (R < ff * 1.2 * disk_mesh.Rout) & ((R > disk_mesh.Rout))
            ff*=1.05
            
        self.params.max_volume = 4.0/3*np.pi * disk_mesh.Rout**3 * (1.2**3-1.0)/ R[ind].shape[0]
        ind = (R > disk_mesh.Rin) & ((R < disk_mesh.Rout))
        self.params.max_volume = self.params.reference_gas_part_mass/dens[ind].min()
        self.params.min_volume = self.params.reference_gas_part_mass/dens[ind].max()

        # Obtain the temperature balance far from the disk
        if (disk.__class__.__name__ == 'disk3d'):
            press_background = press[dens == dens[R >= disk_mesh.Rout].min()].mean()
            dens_background = dens[dens == dens[R >= disk_mesh.Rout].min()].mean()
            self.params.limit_u_below_this_density = dens_background
            self.params.limit_u_below_this_density_to_this_value = press_background / (disk.adiabatic_gamma - 1.0) / dens_background 
        
        # Assign the box size
        self.params.box_size = self.BoxSize

        # Softening parameter
        if (disk.central_particle is False):
            self.params.central_mass = disk.Mcentral
            self.params.softening_central_mass = disk.Mcentral_soft
        else:
            if (STAR_PARTTYPE == 1):
                self.params.softening_type_of_parttype_1 = disk.Mcentral_soft
                self.params.softening_comoving_type_1 = disk.Mcentral_soft
                self.params.softening_max_phys_type_1 = disk.Mcentral_soft
            if (STAR_PARTTYPE == 2):
                self.params.softening_type_of_parttype_2 = disk.Mcentral_soft
                self.params.softening_comoving_type_2 = disk.Mcentral_soft
                self.params.softening_max_phys_type_2 = disk.Mcentral_soft
            if (STAR_PARTTYPE == 3):
                self.params.softening_type_of_parttype_3 = disk.Mcentral_soft
                self.params.softening_comoving_type_3 = disk.Mcentral_soft
                self.params.softening_max_phys_type_3 = disk.Mcentral_soft
            if (STAR_PARTTYPE == 4):
                self.params.softening_type_of_parttype_4 = disk.Mcentral_soft
                self.params.softening_comoving_type_4 = disk.Mcentral_soft
                self.params.softening_max_phys_type_4 = disk.Mcentral_soft

        # Other variable parameters
        if (disk.l < 1.e-2):
            self.params.iso_sound_speed = disk.csnd0
                
    def load(self,R,phi,z,dens,mass,vphi,vr,press,ids,dims=3,particle_type=0,adiabatic_gamma=1.4):

        if (dims == 3):
            X0  = 0.5 * self.BoxSize
            Y0  = 0.5 * self.BoxSize
            Z0  = 0.5 * self.BoxSize
        elif (dims == 2):
            X0  = 0.5 * self.BoxSize
            Y0  = 0.5 * self.BoxSize
            Z0  = 0
            z = np.zeros(len(R))

        x = R * np.cos(phi) + X0
        y = R * np.sin(phi) + Y0
        z = z + Z0

        try:
            vx = vr * np.cos(phi) - vphi * np.sin(phi)
        except TypeError:
            vx = None
        try:
            vy = vr * np.sin(phi) + vphi * np.cos(phi)
        except TypeError:
            vy = None
        try:
            vz = np.zeros(z.shape[0])
        except TypeError:
            vz =  None

            
        if (particle_type == 0):
            if (mass is not None):
                self.gas.mass = mass
            if (dens is not None):
                self.gas.dens = dens
            self.gas.press = press
            self.gas.pos = np.array([x,y,z]).T
            self.gas.vel = np.array([vx,vy,vz]).T
            try:
                self.gas.utherm = press/dens/(adiabatic_gamma - 1)
            except TypeError:
                self.gas.utherm = None
            self.gas.ids = ids
            
        elif (particle_type == STAR_PARTTYPE):
            self.particle.mass = mass
            self.particle.pos = np.array([x,y,z]).T
            self.particle.vel = np.array([vx,vy,vz]).T
            self.particle.ids = ids 
    
    
     
    def assign_primitive_variables(self,disk,disk_mesh):

        if (disk.__class__.__name__ == 'disk3d'):
            return assign_primitive_variables_3d(disk,disk_mesh)

        if (disk.__class__.__name__ == 'disk2d'):
            return assign_primitive_variables_2d(disk,disk_mesh)
        
        '''
        R,phi,z = disk_mesh.create(disk)
        
        x = R*np.cos(phi)
        y = R*np.sin(phi)
        points = np.array([x,y,z]).T
        #print "Voronoi"
        #vor = Voronoi(points)
        
        R1,R2 = min(1e-4,0.9*R.min()),1.5*disk_mesh.Rout
        #obtain density of cells
        dens, radii, midplane_dens = disk.solve_vertical_structure(R,z,R1,R2,disk_mesh.Ncells)
        dens_cut = midplane_dens[-1]
        radii = np.append(radii,R2)
        midplane_dens = np.append(midplane_dens,dens_cut)
        dens[dens < dens_cut] = dens_cut
        midplane_dens[midplane_dens < dens_cut] = dens_cut
        #window_length = 20
        #weights = np.exp(np.linspace(-1., 0., window_length))
        #midplane_dens = np.convolve(midplane_dens,weights/np.sum(weights),mode='same')
        dens0_profile =  interp1d(radii,midplane_dens,kind='linear')#,fill_value='extrapolate')

        
        #evaluate other quantities
        Nvals = 1200 # this number being large can be critical when steep pressure gradients are present
        scale = 'log'
        R1,R2 = 0.99*R.min(),disk_mesh.Rout
        while (True):
            radii, angular_frequency_sq = disk.evaluate_angular_freq_gravity(R1,R2,Nvals=Nvals,scale=scale)
            plt.plot(radii,radii*np.sqrt(angular_frequency_sq))
            plt.plot(radii,1/np.sqrt(radii))
            plt.show()
            _, sound_speed = disk.evaluate_soundspeed(R1,R2,Nvals=Nvals,scale=scale)
            pressure_midplane = dens0_profile(radii) * sound_speed**2
            _,pressure_midplane_gradient =  disk.evaluate_radial_gradient(pressure_midplane,R1,R2,Nvals=Nvals,scale=scale)
            _,soundspeed_sq_gradient =  disk.evaluate_radial_gradient(sound_speed**2,R1,R2,Nvals=Nvals,scale=scale)
            if np.all((angular_frequency_sq + pressure_midplane_gradient/dens0_profile(radii)/radii) > 0):
                break
            else:
                R1,R2 = 1.0001 * R1, 0.999 * R2
                Nvals = int(0.999 * (Nvals-1))
            if (Nvals < 500):
                print "Error: Disk TOO THICK or number of cells TOO LOW to capture rotation curve accurately. Try again"
                exit()
                    
            
        angular_frequency_midplane = np.sqrt(angular_frequency_sq + pressure_midplane_gradient/dens0_profile(radii)/radii)
            
        #update mesh radial limits
        disk_mesh.Rin, disk_mesh.Rout = radii.min(),radii.max()
        
        #interpolate mid-plane quantities
        vphi_profile = interp1d(radii,angular_frequency_midplane*radii,kind='linear')
        soundspeedsq_profile = interp1d(radii,sound_speed**2,kind='linear')
        soundspeedsq_gradient_profile = interp1d(radii,soundspeed_sq_gradient,kind='linear')

        # primitive variables inside the disk
        ind_in = (R > disk_mesh.Rin) & (R < disk_mesh.Rout) & (np.abs(z) < disk_mesh.zmax) 
        vphi, press = np.zeros(R.shape),np.zeros(R.shape)
        vphi[ind_in] = vphi_profile(R[ind_in]) -  soundspeedsq_gradient_profile(R[ind_in]) * np.log(dens[ind_in]/dens0_profile(R[ind_in]))
        press[ind_in] = dens[ind_in] * soundspeedsq_profile(R[ind_in])

        # behavior outside the disk
        ind_out = (R >= disk_mesh.Rout) | (np.abs(z) >= disk_mesh.zmax) | (R <= disk_mesh.Rin) 
        vphi[ind_out] = 0
        dens[ind_out] = dens_cut/1000000
        press_cut = dens_cut * soundspeed(disk_mesh.Rout,disk.csnd0,disk.l,disk.csndR0)**2
        press[ind_out] = press_cut

        ind = R < disk_mesh.Rin 
        vphi[ind] = vphi[ind]*np.exp(-(disk_mesh.Rin-R[ind])**2/R[ind]**2)
        dens[ind] = dens_cut/1000000

        # outside the disk proper, we want a hot, dilute medium that is also ~stationary
        ind = dens < dens_cut/100
        press[ind] = press_cut
        vphi[ind] = 0.0

        
        vr = np.zeros(R.shape)
        ids = np.arange(1,R.shape[0]+1,1)
        '''
        
        #return R,phi,z,dens,vphi,vr,press,ids


    def incline(self,theta,phi,disk_mesh):
        costheta,sintheta = np.cos(theta*np.pi/180.0),np.sin(theta*np.pi/180.0)
        cosphi,sinphi = np.cos(phi*np.pi/180.0),np.sin(phi*np.pi/180.0)


        self.gas.pos[:,0]-= 0.5 * self.BoxSize
        self.gas.pos[:,1]-= 0.5 * self.BoxSize
        self.gas.pos[:,2]-= 0.5 * self.BoxSize
        
        R = np.sqrt(self.gas.pos[:,0]**2+self.gas.pos[:,1]**2+self.gas.pos[:,2]**2)
        ind = R < 1.5 * disk_mesh.Rout 
        
        self.gas.pos[ind,1],self.gas.pos[ind,2] = costheta * (self.gas.pos[ind,1]) - sintheta * self.gas.pos[ind,2],\
                                                  sintheta * self.gas.pos[ind,1] + costheta * self.gas.pos[ind,2]
        self.gas.pos[ind,0],self.gas.pos[ind,1] = cosphi * self.gas.pos[ind,0] - sinphi * self.gas.pos[ind,1], \
                                                  sinphi * self.gas.pos[ind,0] + cosphi * self.gas.pos[ind,1]
        
        self.gas.vel[ind,1],self.gas.vel[ind,2] = costheta * self.gas.vel[ind,1] - sintheta * self.gas.vel[ind,2],\
                                                  sintheta * self.gas.vel[ind,1] + costheta * self.gas.vel[ind,2]
        self.gas.vel[ind,0],self.gas.vel[ind,1] = cosphi * self.gas.vel[ind,0] - sinphi * self.gas.vel[ind,1], \
                                                  sinphi * self.gas.vel[ind,0] + cosphi * self.gas.vel[ind,1]
        
        self.gas.pos[:,0]+= 0.5 * self.BoxSize
        self.gas.pos[:,1]+= 0.5 * self.BoxSize
        self.gas.pos[:,2]+= 0.5 * self.BoxSize

        # Make sure there are no mesh-generating points outside the box after rotation
        ind = (self.gas.pos[:,0] > 0) & (self.gas.pos[:,0] < self.BoxSize) &\
              (self.gas.pos[:,1] > 0) & (self.gas.pos[:,1] < self.BoxSize) &\
              (self.gas.pos[:,2] > 0) & (self.gas.pos[:,2] < self.BoxSize) 
        self.extract(ind)
        
        
    def extract(self,index):
        self.gas.pos=self.gas.pos[index,:]
        self.gas.vel=self.gas.vel[index,:]
        self.gas.dens=self.gas.dens[index]
        self.gas.press=self.gas.press[index]
        if (self.gas.utherm is not None):
            self.gas.utherm=self.gas.utherm[index]
        if (self.gas.mass is not None):
            self.gas.mass=self.gas.mass[index] 
        self.gas.ids=self.gas.ids[index]

    def append(self,snapshot):
        self.gas.pos=np.concatenate([self.gas.pos,snapshot.gas.pos],axis=0)
        self.gas.vel=np.concatenate([self.gas.vel,snapshot.gas.vel],axis=0)
        self.gas.dens=np.append(self.gas.dens,snapshot.gas.dens)
        if (self.gas.utherm is not None):
            self.gas.utherm=np.append(self.gas.utherm,snapshot.gas.utherm)
        self.gas.ids=np.append(self.gas.ids,snapshot.gas.ids)
        self.gas.ids[self.gas.ids > 0] = np.arange(1,1+self.gas.ids[self.gas.ids > 0].shape[0])

    def load_particles(self,part_data):
        self.particle.pos  = np.add(part_data.pos,np.array([0.5 * self.BoxSize,0.5 * self.BoxSize,0.5 * self.BoxSize]))
        self.particle.vel  = part_data.vel
        self.particle.mass = part_data.mass
        self.particle.ids  = part_data.ids

    def add_one_particle(self, x = 0.0, y = 0.0, z = 0.0, vx = 0.0, vy = 0.0, vz = 0.0, m = 0.0,
                         a = None, e = None, I = None, g = None, h = None, l = None):

        try:
            idmax =  max(self.gas.ids.max(),self.particle.ids.max())
        except ValueError:
            idmax = self.gas.ids.max()
        self.particle.add_particle(x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,m=m,a=a,e=e,I=I,g=g,h=h,l=l,
                                   ID=idmax+1)
        
        # Correct locations
        self.particle.center_of_mass_frame()
        self.particle.pos  = np.add(self.particle.pos,np.array([0.5 * self.BoxSize,0.5 * self.BoxSize,0.5 * self.BoxSize]))
        
    def write_snapshot(self,disk,disk_mesh,filename="./disk.dat.hdf5",time=0, \
                       relax_density_in_input = False):
        
        if not (self.gas.pos is None):
            Ngas = self.gas.pos.shape[0]
        else:
            Ngas = 0
            
        if not (self.particle.pos is None):
            Nparticle = self.particle.pos.shape[0]
        else:
            Nparticle = 0
            
        f=ws.openfile(filename)
        npart=np.array([Ngas,0,0,0,0,0], dtype="uint32")
        npart[STAR_PARTTYPE] = Nparticle
        massarr=np.array([0,0,0,0,0,0], dtype="float64")
        header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr, time=time,
                              boxsize=self.BoxSize, double = np.array([1], dtype="int32"))
        
        ws.writeheader(f, header)
        ws.write_block(f, "POS ", 0, self.gas.pos)
        ws.write_block(f, "VEL ", 0, self.gas.vel)
        if (relax_density_in_input):
            ws.write_block(f, "MASS", 0, self.gas.dens)
        else:
            if (self.gas.mass is not None):
                ws.write_block(f, "MASS", 0, self.gas.mass)
            ws.write_block(f, "RHO ", 0, self.gas.dens)
        if (self.gas.utherm is not None):
            ws.write_block(f, "U   ", 0, self.gas.utherm)
        ws.write_block(f, "ID  ", 0, self.gas.ids)

        if (Nparticle > 0):
            ws.write_block(f, "POS ", STAR_PARTTYPE, self.particle.pos)
            ws.write_block(f, "VEL ", STAR_PARTTYPE, self.particle.vel)
            ws.write_block(f, "MASS", STAR_PARTTYPE, self.particle.mass)
            ws.write_block(f, "ID  ", STAR_PARTTYPE, self.particle.ids)

        ws.closefile(f)
        
    def write_parameter_file(self,disk,disk_mesh,filename="./param.txt",time=0):
        self.params.write(filename)
        






def assign_primitive_variables_2d(disk,disk_mesh):

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

        # Check if there are non-axisymmetric perturbations
        if (disk.density_perturbation_function is not None):
            index = phi <= np.pi
            dens[index]+= np.vectorize(disk.density_perturbation_function)(R[index],phi[index])
            dens[np.invert(index)]+= np.vectorize(disk.density_perturbation_function)(R[np.invert(index)],
                                                                                     2 * np.pi-phi[np.invert(index)])
            
        
        #cell ids
        ids = np.arange(1,R.shape[0]+1,1) # Default ID assignment

        #check for boundaries, first inside the boundary
        if (disk_mesh.N_inner_boundary_rings > 0):
            ind_inner_inside = (R > disk_mesh.Rin) & (R < (disk_mesh.Rin + disk_mesh.N_inner_boundary_rings * disk_mesh.deltaRin))
        else:
            ind_inner_inside = np.repeat(0,R.shape[0]).astype(bool)
        if (disk_mesh.N_outer_boundary_rings > 0):                
            ind_outer_inside = (R < disk_mesh.Rout) & (R > (disk_mesh.Rout - disk_mesh.N_outer_boundary_rings * disk_mesh.deltaRout))
        else:
            ind_outer_inside = np.repeat(0,R.shape[0]).astype(bool)
        ids[ind_inner_inside+ind_outer_inside] = -1

        
        #now outside the boundary
        if (disk_mesh.N_inner_boundary_rings > 0):
            ind_inner_outside = (R < disk_mesh.Rin) & (R > (disk_mesh.Rin - disk_mesh.N_inner_boundary_rings * disk_mesh.deltaRin))
        else:
            ind_inner_outside = np.repeat(0,R.shape[0]).astype(bool)
        if (disk_mesh.N_outer_boundary_rings > 0):    
            ind_outer_outside = (R > disk_mesh.Rout) & (R < (disk_mesh.Rout + disk_mesh.N_outer_boundary_rings * disk_mesh.deltaRout))
        else:
            ind_outer_outside = np.repeat(0,R.shape[0]).astype(bool)
        ids[ind_inner_outside+ind_outer_outside] = -2

        if (disk.sigma_back is not None):
            dens[ind_inner_outside+ind_outer_outside] = disk.sigma_back
            vr[dens <= disk.sigma_back] = 0
            vphi[dens <= disk.sigma_back] = 0
            press[dens <= disk.sigma_back] = press[dens <= disk.sigma_back].min()

            
        #or buffer cells (only if there are boundaries at the interface)
        ind_inner = (R < (disk_mesh.Rin - disk_mesh.N_inner_boundary_rings * disk_mesh.deltaRin))
        ind_outer = (R > (disk_mesh.Rout + disk_mesh.N_outer_boundary_rings * disk_mesh.deltaRout))
        if (disk_mesh.N_inner_boundary_rings > 0):  ids[ind_inner] = range(-3,-3-len(ids[ind_inner]),-1)
        if (disk_mesh.N_outer_boundary_rings > 0):  ids[ind_outer] = range(-3,-3-len(ids[ind_outer]),-1)

        
        if (disk.sigma_back is not None):
            dens[ind_inner+ind_outer] = disk.sigma_back
            vr[dens <= disk.sigma_back] = 0
            vphi[dens <= disk.sigma_back] = 0
            press[dens <= disk.sigma_back] = press[dens <= disk.sigma_back].min()


        vphi[ids < -2] = 0
        vr[ids < -2] = 0

        z = np.zeros(dens.shape[0])
        
        return R,phi,z,dens,vphi,vr,press,ids


    

def assign_primitive_variables_3d(disk,disk_mesh):

    R, phi, z = disk_mesh.create(disk)


    x = R*np.cos(phi)
    y = R*np.sin(phi)
    points = np.array([x,y,z]).T
    #print "Voronoi"
    #vor = Voronoi(points)
        
    R1,R2 = min(1e-4,0.9*R.min()),1.5*disk_mesh.Rout
    #obtain density of cells
    dens, radii, midplane_dens = disk.solve_vertical_structure(R,z,R1,R2,disk_mesh.Ncells)
    dens_cut = max(midplane_dens[-1],midplane_dens[midplane_dens > 0].min())/100

    radii = np.append(radii,R2)
    midplane_dens = np.append(midplane_dens,dens_cut)
    dens[dens < dens_cut] = dens_cut
    midplane_dens[midplane_dens < dens_cut] = dens_cut
    print "Density cutoff is:", dens_cut
    #window_length = 20
    #weights = np.exp(np.linspace(-1., 0., window_length))
    #midplane_dens = np.convolve(midplane_dens,weights/np.sum(weights),mode='same')
    dens0_profile =  interp1d(radii,midplane_dens,kind='linear',fill_value='extrapolate')
    
    #evaluate other quantities
    Nvals = 1200 # this number being large can be critical when steep pressure gradients are present

    if (disk_mesh.NR is not None):
        Nvals = disk_mesh.NR        
        scale = 'log'
        R1, R2 = disk_mesh.Rin,disk_mesh.Rout
    else:
        Nvals = 1200
        scale = 'log'
        R1,R2 = 0.99*R.min(),disk_mesh.Rout

    try_count = 0
    Nvals = 1200

    spinner = itertools.cycle(['-', '/', '|', '\\'])

    while (True):
        sys.stdout.write(spinner.next()) 
        sys.stdout.flush()               
        sys.stdout.write('\b')    
        
        radii, angular_frequency_sq = disk.evaluate_angular_freq_gravity(R1,R2,Nvals=Nvals,
                                                                         scale=scale)

        _, sound_speed = disk.evaluate_soundspeed(R1,R2,Nvals=Nvals,scale=scale)
        pressure_midplane = dens0_profile(radii) * sound_speed**2

        _,pressure_midplane_gradient =  disk.evaluate_radial_gradient(pressure_midplane,R1,R2,Nvals=Nvals,
                                                                      scale=scale)
        _,soundspeed_sq_gradient =  disk.evaluate_radial_gradient(sound_speed**2,R1,R2,Nvals=Nvals,
                                                                  scale=scale)
        if np.all((angular_frequency_sq + pressure_midplane_gradient/dens0_profile(radii)/radii) > 0):
            break
        else:
            R1,R2 = 1.0001 * R1, 0.999 * R2
            Nvals = int(0.999 * (Nvals-1))
        if (Nvals < 100):
            print "Error: Disk TOO THICK or number of cells TOO LOW to capture rotation curve accurately. Try again"
            exit()
        try_count+=1

        if (try_count > 30):
            print "We are stuck trying to fix a rotation curve that can turn negative due to steep pressure gradients. Try new profile or more cells."

            
    angular_frequency_midplane = np.sqrt(angular_frequency_sq + pressure_midplane_gradient/dens0_profile(radii)/radii)
            
    #update mesh radial limits
    disk_mesh.Rin, disk_mesh.Rout = radii.min(),radii.max()
        
    #interpolate mid-plane quantities
    vphi_profile = interp1d(radii,angular_frequency_midplane*radii,kind='linear')
    soundspeedsq_profile = interp1d(radii,sound_speed**2,kind='linear')
    soundspeedsq_gradient_profile = interp1d(radii,soundspeed_sq_gradient,kind='linear')

    # primitive variables inside the disk
    ind_in = (R > disk_mesh.Rin) & (R < disk_mesh.Rout) & (np.abs(z) < 1.5 * disk_mesh.zmax) 
    vphi, press = np.zeros(R.shape),np.zeros(R.shape)
    vphi[ind_in] = np.sqrt(vphi_profile(R[ind_in])**2 -  R[ind_in] * soundspeedsq_gradient_profile(R[ind_in]) * np.log(dens[ind_in]/dens0_profile(R[ind_in])))
    press[ind_in] = dens[ind_in] * soundspeedsq_profile(R[ind_in])

    # behavior outside the disk
    ind_out = (R >= disk_mesh.Rout) | (np.abs(z) >= 1.5 * disk_mesh.zmax) | (R <= disk_mesh.Rin) 
    vphi[ind_out] = 0
    dens[ind_out] = dens_cut/10000000
    def soundspeed(R): return disk.csnd0 * (R/disk.csndR0)**(-disk.l*0.5)
    press_cut = dens_cut * soundspeed(disk_mesh.Rout)**2
    press[ind_out] = press_cut
        
    ind = R < disk_mesh.Rin 
    vphi[ind] = vphi[ind]*np.exp(-(disk_mesh.Rin-R[ind])**2/R[ind]**2)
    dens[ind] = dens_cut/1000000
    
    # outside the disk proper, we want a hot, dilute medium that is also ~stationary
    ind = dens < dens_cut/100
    press[ind] = press_cut
    vphi[ind] = 0.0
    
    
    vr = np.zeros(R.shape)
    ids = np.arange(1,R.shape[0]+1,1)

    # Checking enclosed mass
    radial_bins = disk.evaluate_radial_mass_bins(disk_mesh.Rin,disk_mesh.Rout,200)
    bin_inds=np.digitize(R,radial_bins)
    for kk in range(0,radial_bins.shape[0]):
        if (R[bin_inds == kk].shape[0] == 0): continue
        bin_radius = R[bin_inds == kk].mean()
        _, zmvals = disk.evaluate_enclosed_vertical(bin_radius,0,z[bin_inds == kk].max(),Nzvals=300)
    
    print "Done."
    
    return R,phi,z,dens,vphi,vr,press,ids

