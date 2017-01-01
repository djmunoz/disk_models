import numpy as np
import matplotlib.pyplot as plt
import snapHDF5 as ws
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from scipy.spatial import Voronoi
import scipy.integrate as integ

from disk_structure import *
from disk_parameter_files import *
from disk_particles import *

STAR_PARTTYPE = 4

class gas_data():
    def __init__(self,*args,**kwargs):
        self.pos=kwargs.get("pos")
        self.vel=kwargs.get("vel")
        self.dens=kwargs.get("dens")
        self.utherm=kwargs.get("utherm")
        self.ids=kwargs.get("ids")

    


class snapshot():

    def __init__(self,*args,**kwargs):

        self.gas = gas_data()
        self.particle = particle_data()
        
        self.params = paramfile(init_cond_file="./disk.dat")

        self.BoxSize = kwargs.get("BoxSize")
        
    def create(self,disk,disk_mesh):

        self.BoxSize = disk_mesh.BoxSize
        
        # Obtain the primitive quantities for all cells
        R,phi,z,dens,vphi,vr,press,ids = self.assign_primitive_variables(disk,disk_mesh)
        
        # Load them into the snapshot
        self.load(R,phi,z,dens,None,vphi,vr,press,ids,self.BoxSize,adiabatic_gamma=disk.adiabatic_gamma)

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
        press_background = press[dens == dens[R > 1.2 * disk_mesh.Rout].min()].mean()
        dens_background = dens[dens == dens[R > 1.2 * disk_mesh.Rout].min()].mean()
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
                
    def load(self,R,phi,z,dens,mass,vphi,vr,press,ids,BoxSize,particle_type=0,adiabatic_gamma=1.4):
        
        x = R * np.cos(phi) + 0.5 * BoxSize
        y = R * np.sin(phi) + 0.5 * BoxSize
        z = z + 0.5 * BoxSize
        
        vx = vr * np.cos(phi) - vphi * np.sin(phi)
        vy = vr * np.sin(phi) + vphi * np.cos(phi)
        vz = np.zeros(vx.shape[0])

        if (particle_type == 0):
            self.gas.dens = dens
            self.gas.pos = np.array([x,y,z]).T
            self.gas.vel = np.array([vx,vy,vz]).T
            self.gas.utherm = press/self.gas.dens/(adiabatic_gamma - 1)
            self.gas.ids = ids
            
        elif (particle_type == STAR_PARTTYPE):
            self.particle.mass = mass
            self.particle.pos = np.array([x,y,z]).T
            self.particle.vel = np.array([vx,vy,vz]).T
            self.particle.ids = ids 
    
    
     
    def assign_primitive_variables(self,disk,disk_mesh):
        
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
        ind_out = (R >= disk_mesh.Rout) | (np.abs(z) > disk_mesh.zmax) 
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

        
        return R,phi,z,dens,vphi,vr,press,ids


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

    def extract(self,index):
        self.gas.pos=self.gas.pos[index,:]
        self.gas.vel=self.gas.vel[index,:]
        self.gas.dens=self.gas.dens[index]
        self.gas.utherm=self.gas.utherm[index]
        self.gas.ids=self.gas.ids[index]

    def append(self,snapshot):
        self.gas.pos=np.concatenate([self.gas.pos,snapshot.gas.pos],axis=0)
        self.gas.vel=np.concatenate([self.gas.vel,snapshot.gas.vel],axis=0)
        self.gas.dens=np.append(self.gas.dens,snapshot.gas.dens)
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
        
    def write_snapshot(self,disk,disk_mesh,filename="./disk.dat.hdf5",time=0):
        
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
        ws.write_block(f, "MASS", 0, self.gas.dens)
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
        






