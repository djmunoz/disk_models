import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import snapHDF5 as ws
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from scipy.spatial import Voronoi
import scipy.integrate as integ

from disk_structure import *
from disk_parameter_files import *

class snapshot():

    def __init__(self,*args,**kwargs):
        self.pos=kwargs.get("pos")
        self.vel=kwargs.get("vel")
        self.dens=kwargs.get("dens")
        self.utherm=kwargs.get("utherm")
        self.ids=kwargs.get("ids")

        self.params = paramfile(init_cond_file="./disk.dat")
        
    def create(self,disk,disk_mesh):
        
        R,phi,z,dens,vphi,vr,press,ids = self.assign_primitive_variables(disk,disk_mesh)


        self.load(R,phi,z,dens,vphi,vr,press,ids,disk_mesh.BoxSize,disk.adiabatic_gamma)

        # Obtain target masses and allowed volumes
        self.params.reference_gas_part_mass = disk.compute_disk_mass(disk_mesh.Rin,disk_mesh.Rout)/disk_mesh.Ncells
        ind = (R < 1.2 * disk_mesh.Rout) & ((R > disk_mesh.Rout))
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
        self.params.box_size = disk_mesh.BoxSize

        
        
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

        print R.shape, phi.shape, z.shape, R.max(), z.max()
        x = R*np.cos(phi)
        y = R*np.sin(phi)
        points = np.array([x,y,z]).T
        #print "Voronoi"
        #vor = Voronoi(points)
        
        R1,R2 = min(1e-4,0.9*R.min()),1.5*disk_mesh.Rout
        #obtain density of cells
        dens, radii, midplane_dens = disk.solve_vertical_structure(R,z,R1,R2,disk_mesh.Ncells)
        print "hold on there",x.shape,dens.shape,disk_mesh.Rout
        dens_cut = midplane_dens[-1]
        dens[dens < dens_cut] = dens_cut
        midplane_dens[midplane_dens < dens_cut] = dens_cut
        dens0_profile =  interp1d(radii,midplane_dens,kind='linear')

        rr,sig =  disk.evaluate_sigma(R1,R2)
        _,cs0 =  disk.evaluate_soundspeed(R1,R2)
        dens0_guess = sig/np.sqrt(2*np.pi)/(cs0*np.sqrt(rr)*rr)

        
        #evaluate other quantities
        R1,R2 = 0.99*R.min(),disk_mesh.Rout
        Nvals = 1200 # this number can be critical when steep pressure gradients are present
        radii, angular_frequency_sq = disk.evaluate_angular_freq_gravity(R1,R2,Nvals=Nvals)
        _, sound_speed = disk.evaluate_soundspeed(R1,R2,Nvals=Nvals)
        pressure_midplane = dens0_profile(radii) * sound_speed**2
        _,pressure_midplane_gradient =  disk.evaluate_radial_gradient(pressure_midplane,R1,R2,Nvals=Nvals)
        _,soundspeed_sq_gradient =  disk.evaluate_radial_gradient(sound_speed**2,R1,R2,Nvals=Nvals)
        angular_frequency_midplane = np.sqrt(angular_frequency_sq + pressure_midplane_gradient/dens0_profile(radii)/radii)

        plt.plot(radii,angular_frequency_sq)
        plt.plot(radii,pressure_midplane_gradient/dens0_profile(radii)/radii)
        plt.show()
        
        #interpolate mid-plane quantities
        vphi_profile = interp1d(radii,angular_frequency_midplane*radii,kind='linear')
        soundspeedsq_profile = interp1d(radii,sound_speed**2,kind='linear')
        soundspeedsq_gradient_profile = interp1d(radii,soundspeed_sq_gradient,kind='linear')

        #primitive variables
        vphi, press = np.zeros(R.shape),np.zeros(R.shape)
        ind = (R < disk_mesh.Rout) & (np.abs(z) < disk_mesh.zmax) 
        vphi[ind] = vphi_profile(R[ind]) -  soundspeedsq_gradient_profile(R[ind]) * np.log(dens[ind]/dens0_profile(R[ind]))
        press[ind] = dens[ind] * soundspeedsq_profile(R[ind])
        ind = (R >= disk_mesh.Rout) | (np.abs(z) > disk_mesh.zmax) 
        vphi[ind] = 0
        dens[ind] = dens_cut/1000000
        press_cut = dens_cut * soundspeed(disk_mesh.Rout,disk.csnd0,disk.l,disk.csndR0)**2
        press[ind] = press_cut
        ind = R < disk_mesh.Rin 
        vphi[ind] = vphi[ind]*np.exp(-(disk_mesh.Rin-R[ind])**2/R[ind]**2)
        dens[ind] = dens_cut/10000000
        ind = dens < dens_cut/100
        press[ind] = press_cut

        # outside the disk proper, we want a hot, dilute medium
        print "DENS_CUT",dens_cut
        print "PRESS_CUT",press_cut
        print "jaja",press[press < 0],dens[dens <0]
        print press.shape, dens.shape, R.shape,phi.shape, z.shape
        
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


        
    def write_snapshot(self,disk,disk_mesh,filename="./disk.dat.hdf5",time=0):
        
       
        
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
        
    def write_parameter_file(self,disk,disk_mesh,filename="./param.txt",time=0):
        self.params.write(filename)
        






