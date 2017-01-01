##########################################################################
##########################################################################
#ABOUT

##########################################################################
##########################################################################
import sys
import snapHDF5 as ws
import numpy as np
import numpy.random as rd
from scipy.integrate import quad
from scipy.integrate import cumtrapz
import snapHDF5 as ws
import numpy as np
import numpy.random as rd
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.misc import derivative
import matplotlib.pyplot as plt
from math import factorial
import scipy.integrate as integ
import matplotlib.cm as cm
from smooth import *
from scipy.ndimage.filters import convolve1d
from scipy.signal import gaussian
from matplotlib.colors import LogNorm
#import  matplotlib.axes as maxes

#set unit system
UnitMass_in_g            = 1.98892e33 # solar mass
UnitVelocity_in_cm_per_s = 4.74057581e5 # AU per year
UnitLength_in_cm         = 1.49598e13  # AU
G                        = 39.4751488
BOLTZMANN_CGS            = 1.38066e-16
PROTON_MASS_CGS          = 1.6726e-24
SOLAR_EFF_TEMP           = 5780.0

##################
#INPUT PARAMETERS#

#MAIN PARAMETERS
##################
N_gas    =    int(sys.argv[1])
###################
M_star  =     float(sys.argv[2])
##################
M_disk =      float(sys.argv[3])
###################
R_disk =      float(sys.argv[4])
###################
R_cavity=      float(sys.argv[5])
################## Aspect Ratio at R_disk 
h_Rd =         float(sys.argv[6])
################### density profile index
p =           float(sys.argv[7])
###################temperature profile index
l =           float(sys.argv[8])
################### stellar softening
s =           float(sys.argv[9])
################### Box size
Box =         float(sys.argv[10])
################### output ICs filename
ic_filename = sys.argv[11]

#SECONDARY PARAMETERS
##################
star_type = 4
###################
GAMMA       = 1.00001 #1.40000 #adiabatic index
GAMMA_PRIME = 1.00000 #polytropic index
##################Stellar softening
star_soft = s * 2.8
##################
TEMPERATURE_FLOOR = 10    # in Kelvin
##################
NUMBER_DENSITY_FLOOR_CGS =   10  # particles per cc
##################
NUMBER_DENSITY_CUTOFF_CGS =  10 #particles per cc
################## Composition
X, Y, Z = 0.7, 0.28, 0.02
#################
SelfGravity = True
#################
StarAccretion = True
###############
ExternalPotential = False
###############
CentralCavity = True


##################
#COMPUTATIONAL PARAMETERS#
##################
#Mesh generation
add_background = True
relax_density = True
output_images = False
load_mesh = False
double_precision =  False
output_scaleheight = True
output_velocity_field = True
#
seed     = 42
R_max    = 6.5 * R_disk # maximum sampling radius
if (N_gas < 50000): R_bins = 100
elif (N_gas < 100000): R_bins = 200
elif (N_gas < 200000): R_bins = 400
elif (N_gas < 600000): R_bins   = 600
else : R_bins = 1000
R_cut    = 5.0 * R_disk
scale_height_cut    = 4.0 #cut in the z direction after a number of scale heights
zeta_bins   = 1200
zeta_max = 12.0
z_max = h_Rd * R_cut * 10

dens_soft = 0.08 * R_disk
temp_soft = 1.0 * star_soft

min_cells_per_bin = 10

if (load_mesh):
  import readsnapHDF5 as rs
  mesh_file = "./mesh"
############################################################################################################
BOX_HEIGHT = Box
BOX_WIDTH = Box

#Interpolation parameters
INTERPOL_BINS  = R_bins
INTERPOL_R_MAX = R_max       #maximum gas sampling radius 
INTERPOL_ZETA_BINS = zeta_bins
INTERPOL_ZETA_MAX = zeta_max
INTERPOL_Z_MIN=  zeta_max*1e-4
############################################################################################################
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
#######################################################################################################
#initialize some variables
MassCorrection=1.0
SofteningCorrection=1.0
TruncationCorrection = 1.0
R_min = 0.01

UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitEnergy_in_cgs = UnitMass_in_g * UnitLength_in_cm**2 / UnitTime_in_s**2

#MEAN MOLECULAR WEIGHT
meanweight = 1.0/(0.5*X + Y/4.002602 + 0.5/17.0*Z)

DENSITY_FLOOR = NUMBER_DENSITY_FLOOR_CGS * meanweight * PROTON_MASS_CGS   \
                / UnitMass_in_g * UnitLength_in_cm**3
DENSITY_CUTOFF = NUMBER_DENSITY_CUTOFF_CGS * meanweight * PROTON_MASS_CGS   \
                / UnitMass_in_g * UnitLength_in_cm**3
UTHERM_FLOOR = TEMPERATURE_FLOOR /(GAMMA - 1) \
               * BOLTZMANN_CGS / meanweight / PROTON_MASS_CGS \
               / UnitEnergy_in_cgs * UnitMass_in_g
SOUNDSPEED_SQUARED_FLOOR = UTHERM_FLOOR*(GAMMA-1)*GAMMA_PRIME

SURFACE_DENSITY_FLOOR = DENSITY_FLOOR *2*z_max
#######################################################################################################
#functions

#Aspect ratio at R_disk
Rd_temp = h_Rd**2 * UnitEnergy_in_cgs / UnitMass_in_g * PROTON_MASS_CGS / BOLTZMANN_CGS \
	  * meanweight * G * M_star / R_disk
h_0 = np.sqrt(R_disk * Rd_temp/ G / M_star /meanweight /PROTON_MASS_CGS * BOLTZMANN_CGS \
              / UnitEnergy_in_cgs * UnitMass_in_g)


def GasSigmaAnalytic(R):
  return MassCorrection *  SofteningCorrection * TruncationCorrection *\
         (2.0 - p) * M_disk/2/np.pi/R_disk**2 * (R_disk/R)**p  * np.exp(-(R/R_disk)**(2.0-p))

def GasSigma(R):
  if (not CentralCavity):
    if (not StarAccretion):
      return MassCorrection *  SofteningCorrection * TruncationCorrection *\
          (2.0 - p) * M_disk/2/np.pi/R_disk**2 * R_disk**p * SplineProfile(R,dens_soft)**p \
          * np.exp(-(np.abs(R)/R_disk)**(2.0-p))
    else:
      return MassCorrection *  SofteningCorrection * TruncationCorrection *\
          (2.0 - p) * M_disk/2/np.pi/R_disk**2 * R_disk**p * (SplineDerivative(R,dens_soft)* R * R)**p \
          * np.exp(-(np.abs(R)/R_disk)**(2.0-p))
  else:
    return MassCorrection *  SofteningCorrection * TruncationCorrection *\
        ((2.0 - p) * M_disk/2/np.pi/R_disk**2 * R_disk**p * (SplineDerivative(R,R_cavity)* R * R)**p * np.exp(-(np.abs(R)/R_disk)**(2.0-p)) + SURFACE_DENSITY_FLOOR)

def GasMass(R):
  def GasMass_Integrand(R): return GasSigma(R) * R * 2 * np.pi
  return quad(GasMass_Integrand,0.0,R)[0]


def GasSoundSpeedSquared(R):
  cs2 =  h_0**2 * G * M_star / R_disk * R_disk**l * SplineProfile(R,temp_soft)**l
  if (cs2 >= SOUNDSPEED_SQUARED_FLOOR):
    return cs2
  else:
   return SOUNDSPEED_SQUARED_FLOOR

def GasSoundSpeedSquaredAnalytic(R):
  return h_0**2 * G * M_star / R_disk * (R / R_disk)**(-l)

def GasTemperature(R):
  return GasSoundSpeedSquared(R) / GAMMA_PRIME * meanweight * PROTON_MASS_CGS / BOLTZMANN_CGS \
         * UnitEnergy_in_cgs / UnitMass_in_g

def VerticalGasMass_Integrand(zeta): 
  return  np.exp(-0.5 * zeta **2)

def KeplerianVerticalPotential(R,z):
  return - G*M_star * (SplineProfile(np.sqrt(R*R + z*z),star_soft) - SplineProfile(R,star_soft))

def OmegaKeplerSquared(R,z):
  return G * M_star * SplineDerivative(np.sqrt(R*R + z*z),star_soft) 

def OmegaSquaredAnalytic(R,z):
  return G * M_star / R/  R  /R *(1 - 1.5 *z*z/R/R) - GasSoundSpeedSquaredAnalytic(R)/R/R *( \
    (l + p) + (2.0 - p)*(R/R_disk)**(2.0-p) - 0.5*(3.0-l)*(z*z/GasScaleHeightAnalytic(R)/GasScaleHeightAnalytic(R)-1.0))

def GasScaleHeightAnalytic(R):
  return h_0 * (R/R_disk)**(-l*0.5+0.5) * R 

def GasScaleHeight(R):
  return np.sqrt(GasSoundSpeedSquared(R)/OmegaKeplerSquared(R,0.0))

def GasRhoAnalytic(R,z):
  return GasSigmaAnalytic(R)/np.sqrt(2.0*np.pi) / GasScaleHeightAnalytic(R) * \
         np.exp(-0.5 * z * z/GasScaleHeightAnalytic(R)/GasScaleHeightAnalytic(R))

def VerticalDensity(rho,z,R):
  drho_dz = - rho* G*M_star * SplineDerivative(np.sqrt(R*R + z*z),star_soft) * z /  GasSoundSpeedSquared(R)
  return drho_dz 

def EnthalpyGradientAnalytic(R,z):
  return -GasSoundSpeedSquaredAnalytic(R) / R * \
         ( l + p + (2.0 - p)*(R / R_disk)**(2.0-p) - 0.5*(-l + 3)* \
           ( z*z/GasScaleHeightAnalytic(R)/GasScaleHeightAnalytic(R) - 1))

def DiskToomreQ(R):
  return np.sqrt(GasSoundSpeedSquared(R) *  OmegaKeplerSquared(R,0.0)) / \
         (np.pi * G * GasSigma(R))


############################################################################################################
#mass correction due to singularity softening
SofteningCorrection=M_disk/GasMass(np.inf)
#mass correction due to finite sampling
MassCorrection=GasMass(np.inf)/GasMass(R_max)

#set seed
rd.seed(seed)
print "\n Setting up Lynden-Bell - Pringle Disk profile with ",N_gas," gas particles \n"
print "\n-STARTING-\n"

#vectorize functions
print "Vectorizing functions..."
vecSigma=np.vectorize(GasSigma)
vecSigmaAnalytic=np.vectorize(GasSigmaAnalytic)
vecGasMass=np.vectorize(GasMass)
vecGasRhoAnalytic=np.vectorize(GasRhoAnalytic)
vecOmegaKeplerSquared=np.vectorize(OmegaKeplerSquared)
vecOmegaSquaredAnalytic=np.vectorize(OmegaSquaredAnalytic)
vecSoundSpeedSquared=np.vectorize(GasSoundSpeedSquared)
vecGasTemperature=np.vectorize(GasTemperature)
vecSoundSpeedSquaredAnalytic=np.vectorize(GasSoundSpeedSquaredAnalytic)
vecGasScaleHeightAnalytic = np.vectorize(GasScaleHeightAnalytic)
vecGasScaleHeight = np.vectorize(GasScaleHeight)
vecEnthalpyGradientAnalytic = np.vectorize(EnthalpyGradientAnalytic)
vecKeplerianVerticalPotential = np.vectorize(KeplerianVerticalPotential)
vecDerivative =  np.vectorize(derivative)
vecDiskToomreQ = np.vectorize(DiskToomreQ)


print "done."
###############################
print "finding minimum sampling radius..."
#R_min = 0.1* np.sqrt(M_disk/N_gas/GasMass(dens_soft))*dens_soft
INTERPOL_R_MIN = R_min
print "   RMIN",R_min
print "done."


print "\n2-D structure..."
print "    Inverting/Interpolating functions..."
#invert function: GasMass^-1 = GasRadius 
radial_bins=np.exp(np.arange(INTERPOL_BINS) * np.log(INTERPOL_R_MAX/INTERPOL_R_MIN)/(INTERPOL_BINS) + np.log(INTERPOL_R_MIN))
mass_bins_gas=vecGasMass(radial_bins)
MasstoRadius=InterpolatedUnivariateSpline(mass_bins_gas, radial_bins,k=1)

#we redo the radial bins based on the cumulative mass function
bin_concentration = 0.5
bin_break = 0.1
radial_bins1= MasstoRadius(np.arange(0.0,bin_break,bin_break/np.int(INTERPOL_BINS*bin_concentration))*mass_bins_gas.max())
radial_bins2= MasstoRadius(np.linspace(bin_break,1.0,(INTERPOL_BINS - np.int(INTERPOL_BINS*bin_concentration)))*mass_bins_gas.max())


radial_bins=np.append(radial_bins1,radial_bins2)
radial_bins=MasstoRadius(np.linspace(0.0,1.0,INTERPOL_BINS)*mass_bins_gas.max())
mass_bins_gas=vecGasMass(radial_bins)
MasstoRadius=InterpolatedUnivariateSpline(mass_bins_gas, radial_bins,k=3)

if not (load_mesh):
  print "    Inversion sampling..."
  #generate random positions gas
  #fully random
  radius_gas= MasstoRadius(rd.random_sample(N_gas)*mass_bins_gas.max())
  print "done."
else:
  #read-in, positions of gas cells
  print "        loading mesh..."
  pos = rs.read_block(mesh_file,"POS ",parttype=0)
  pos[:,0],pos[:,1], pos[:,2] = (pos[:,0]- 0.5 * BOX_WIDTH), (pos[:,1]- 0.5 * BOX_WIDTH), (pos[:,2]- 0.5 * BOX_HEIGHT)
  radius_gas = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
  vertical_gas = pos[:,2]

  ind = (np.abs(vertical_gas) < z_max) & (radius_gas < R_max)
  radius_gas = radius_gas[ind]
  vertical_gas = vertical_gas[ind]
  
  print "done"
  

#first we bin our newly sampled list of radii
bin_inds=np.digitize(radius_gas,radial_bins)
#check for empty or underpopulated bins:
for k in range(0,INTERPOL_BINS-1):
  N_in_bin = radius_gas[bin_inds == k].shape[0]
  if (N_in_bin < min_cells_per_bin):
    delta_cells  = min_cells_per_bin - N_in_bin
    radius_gas=np.append(radius_gas,rd.random_sample(delta_cells)*(radial_bins[k+1]-radial_bins[k])
                         + radial_bins[k])
#then rebin
bin_inds=np.digitize(radius_gas,radial_bins)

#initialize some variables and functions
radius = 0.0
radius_prev = 0.0
radius_prev_prev = 0.0
Pressure_Gradient=[]
R_gas,z_gas,press_gas,dens_gas = [], [], [],[]
Vel_Correction = []

if (add_background) & (not load_mesh):
  R_back,z_back = [], []
  dens_back, press_back = [],[]
  Vel_Corr_back = []
  mass_back = []
  max_z_back = []
  R_back_bins = []  
local_scale_height = []
effective_gas_sigma = []
radius_per_bin,density_in_plane,pressure_in_plane = [],[],[]
selfgravity_vcirc_in_plane = []

#and we loop over radial bins
N_in_bin=0
count = 0
max_vert_iter = 0

TruncationCorrection = GasMass(R_max)/GasMass(R_cut)

ind = vecDiskToomreQ(radial_bins) == (vecDiskToomreQ(radial_bins).min())
print "\n Minimum Toomre-Q value",vecDiskToomreQ(radial_bins).min(),\
      "at R=",radial_bins[ind][0],"AU"
ind = vecSigma(radial_bins) == vecSigma(radial_bins).max()
print "\n Maximum Surface Density",\
       vecSigma(radial_bins).max()*UnitMass_in_g/UnitLength_in_cm**2,"g cm^-2",\
      "at R=",radial_bins[ind][0],"AU"

#plt.loglog(radial_bins,vecDiskToomreQ(radial_bins))
#plt.show()


radius_bin_in = 0.0
radius_bin_out = R_max 

print "\nIntegrating vertical structure..."
print "   Looping over ",R_bins," radial bins..."

for k in range(0,INTERPOL_BINS):
  

  ############################################################
  #Some initialization steps
  if (k%50 ==0):
    print "...\n",
  if (k%100 ==0):
    print "...",'%4.1f' % (100.0*k/R_bins),"% completed"

  ######## number of particles in each bin
  N_in_bin = radius_gas[bin_inds == k].shape[0]
  if (N_in_bin > 0):
    count = count + 1
  
  ########## inner and outer edges of the bin
  if (k < INTERPOL_BINS - 1):
    radius_bin_in = radial_bins[k]
    radius_bin_out = radial_bins[k+1]
  else:
    radius_bin_in = radial_bins[k]
    radius_bin_out = R_max 

  ########## mean radius of the bin
  radius = 0.5*(radius_bin_in + radius_bin_out)

  ########## width of the bin and mass per bin
  if (k > 0):
    delta_bin = radius_bin_out - radius_bin_in
    mass_in_bin = vecGasMass(radius_bin_out) - vecGasMass(radius_bin_in)
  else:
    delta_bin = radial_bins[k]
    mass_in_bin = vecGasMass(radial_bins[k])


  #if the bin is empty, we ignore it
  if (N_in_bin ==0): 
    continue


  ##############################################################
  #Now we start integrating...
  INTERPOL_ZETA_BINS = zeta_bins
  INTERPOL_ZETA_MAX = zeta_max
  
  if not (SelfGravity): #no self-gravity
    def integrand(z): return np.exp(-vecKeplerianVerticalPotential(radius,z)/GasSoundSpeedSquared(radius))
    VertProfileNorm = vecSigma(radius)/(2.0*quad(integrand,0.0,np.inf)[0])

    def density_interp(z_vals):
      return VertProfileNorm * np.exp(-vecKeplerianVerticalPotential(radius,z_vals)/vecSoundSpeedSquared(radius))

    def local_density(R_vals,z_vals):
      return VertProfileNorm * np.exp(-vecKeplerianVerticalPotential(R_vals,z_vals)/vecSoundSpeedSquared(R_vals))
    
    scale_height_guess = vecGasScaleHeight(radius)

    vertical_bins= np.append(0.0,(np.arange(0.5/INTERPOL_ZETA_BINS,1.0,1.0/INTERPOL_ZETA_BINS) * INTERPOL_ZETA_MAX)*scale_height_guess)

    rho_plane, rho_plane_old = VertProfileNorm,1.0e-40
    
  else: #if we are considering self-gravity

    if (M_disk < 0.1 * M_star):  
      def integrand(z): return np.exp(-vecKeplerianVerticalPotential(radius,z)/GasSoundSpeedSquared(radius))
      VertProfileNorm = vecSigma(radius)/(2.0*quad(integrand,0.0,np.inf)[0])
      scale_height_guess = vecGasScaleHeight(radius)
    else:
      VertProfileNorm = vecSigma(radius)**2/GasSoundSpeedSquared(radius) / 2.0 * np.pi * G
      if (radius < 2*star_soft): scale_height_guess = vecGasScaleHeight(radius)
      else: scale_height_guess = GasSoundSpeedSquared(radius)/np.sqrt(2.0 * np.pi * G * VertProfileNorm)
 
    #iterate vertical structure for self-gravity###

    def VerticalPotential(Phi,z,R,rho0):
      dPhiGrad_dz =  rho0 * G* 4 * np.pi * np.exp(-(KeplerianVerticalPotential(R,z) + Phi[1]) / GasSoundSpeedSquared(R))
      dPhi_dz = Phi[0]
      return [dPhiGrad_dz,dPhi_dz]
    
    rho_plane, rho_plane_old = VertProfileNorm,1.0e-40 

    iter = 0
  
    while(True) : #CRUCIAL CALCULATIONS FOR SELF-GRAVITATING DISKS
      if (iter > max_vert_iter): max_vert_iter = iter
      
      tol = 1.0e-8
  
      while (True):
        vertical_bins= np.append(0.0,
                                 (np.arange(0.5/INTERPOL_ZETA_BINS,1.0,1.0/INTERPOL_ZETA_BINS) * INTERPOL_ZETA_MAX)*scale_height_guess)
        min_step = (vertical_bins[1]-vertical_bins[0])/100.0
        soln=integ.odeint(VerticalPotential,[0.0,0.0],vertical_bins,args=(radius,rho_plane),rtol=tol,
                          mxords=15,hmin=min_step,printmessg=False)[:,1]
  
        density = rho_plane*np.exp(-(vecKeplerianVerticalPotential(radius,vertical_bins)+soln)/ \
                                   GasSoundSpeedSquared(radius))
        Phi_interp = InterpolatedUnivariateSpline(vertical_bins, soln,k=3)

        if(density[-1] < DENSITY_CUTOFF): break 
        
        INTERPOL_ZETA_MAX = INTERPOL_ZETA_MAX*1.2
        INTERPOL_ZETA_BINS = INTERPOL_ZETA_BINS* 1.2
             
      def vertical_integrand(z):
        return np.exp(-(KeplerianVerticalPotential(radius,z)+Phi_interp(z))/GasSoundSpeedSquared(radius))
      rho_plane = vecSigma(radius) / (2.0*quad(vertical_integrand,0.0,vertical_bins.max())[0])

            
      abstol,reltol = 1.0e-8,1.0e-5
      abserr,relerr = np.abs(rho_plane_old-rho_plane),np.abs(rho_plane_old-rho_plane)/np.abs(rho_plane_old)

      if (np.abs(rho_plane) > abstol/reltol):
        if (abserr < abstol): break
      else:
        if (relerr < reltol): break
          
    
      rho_plane_old = rho_plane
      iter = iter + 1
           
    INTERPOL_ZETA_BINS = zeta_bins
    INTERPOL_ZETA_MAX = zeta_max
    ##############################
    
    density_interp=InterpolatedUnivariateSpline(vertical_bins,density, k=3)
    
    #density function for R-z plane
    def local_density(R_vals,z_vals):
      return density_interp(z_vals) #* vecSigma(R_vals)/vecSigma(radius)

  ############ end of self-gravity
  
  #local scale-height
  def variance_integrand(z): return density_interp(z)*z**2
  scale = np.sqrt(quad(variance_integrand,0.0,vertical_bins.max())[0]/quad(density_interp,0.0,vertical_bins.max())[0])
  vertical_N_in_bin = N_in_bin * delta_bin /2/np.pi/(scale/radius)
  
  cumulative = np.append(0.0, cumtrapz(density_interp(vertical_bins),vertical_bins))
  CumulInterp = InterpolatedUnivariateSpline(vertical_bins,cumulative,k=3)
  effective_gas_sigma = np.append(effective_gas_sigma,2.0*cumulative.max())
      
  #pressure function for R-z plane
  def local_pressure(R_vals,z_vals):
    return density_interp(z_vals) * vecSoundSpeedSquared(R_vals)

   
  # only do Monte-Carlo sampling if this bin is not empty
  if (N_in_bin != 0):
    MasstoZ=interp1d(cumulative,vertical_bins,bounds_error=False,fill_value=vertical_bins.max(),kind='linear')
    #MasstoZ=InterpolatedUnivariateSpline(cumulative,vertical_bins,k=2)

    if not (load_mesh):
      thickness_range = 1.0 - 0.8*(M_disk/N_gas)/mass_in_bin
      z_gas_at_R=np.append(0.0,MasstoZ(rd.random_sample(N_in_bin-1)* cumulative.max()*thickness_range)) * \
                  (np.round(rd.random_sample(N_in_bin))*2 - 1.0) 
        
      if (z_gas_at_R.shape[0] == 1): z_gas_at_R=np.array([0.0])


    else:
      z_gas_at_R = vertical_gas[bin_inds == k]

    dens_gas_at_R = local_density(radius_gas[bin_inds == k],np.abs(z_gas_at_R))
    press_gas_at_R= local_pressure(radius_gas[bin_inds == k],np.abs(z_gas_at_R))

    leftover_mass = (cumulative.max() - CumulInterp(np.abs(z_gas_at_R).max()))/cumulative.max()
    rescaling_factor = cumulative.max()/CumulInterp(np.abs(z_gas_at_R).max())
    dens_gas_at_R[:] = dens_gas_at_R[:] * rescaling_factor
    press_gas_at_R[:] = press_gas_at_R[:] * rescaling_factor
    


    ind = z_gas_at_R < scale*scale_height_cut
    z_gas=np.append(z_gas,z_gas_at_R[ind])
    press_gas=np.append(press_gas,press_gas_at_R[ind])
    dens_gas = np.append(dens_gas,dens_gas_at_R[ind])
    R_gas = np.append(R_gas,radius_gas[bin_inds == k][ind])
 
    local_scale_height = np.append(local_scale_height,scale)
    radius_per_bin = np.append(radius_per_bin,radius)
    density_in_plane = np.append(density_in_plane,local_density(radius,0.0))
    pressure_in_plane = np.append(pressure_in_plane,local_pressure(radius,0.0))

   
    #Now we calculate the vertical correction to the rotation profile
    Vel_Correction = np.append(Vel_Correction,
                               derivative(vecSoundSpeedSquared,radius_gas[bin_inds == k][ind])*np.log(dens_gas_at_R[ind]/rho_plane) * radius_gas[bin_inds == k][ind])

    #if (k == 292):
    #  plt.loglog(np.linspace(0,vertical_bins.max(),300),local_density(radius,np.linspace(0,vertical_bins.max(),300)),'r-')
    #  #plt.plot(np.arange(0,zeta_max*GasScaleHeightAnalytic(radius),0.003),vecGasRhoAnalytic(radius,np.arange(0,zeta_max*GasScaleHeightAnalytic(radius),0.003)),'gs',ms=3.0)
    #  #plt.plot(z_gas_at_R,dens_gas_at_R,'Dg',ms=2.0)
    #  plt.loglog(np.abs(z_gas_at_R),density_interp(np.abs(z_gas_at_R)),'Db',ms=4.0,mew=0.0)
    #  plt.show()

    if(add_background) & (not load_mesh):
      first = MasstoZ((1.0-leftover_mass)*cumulative.max())
      first = max(np.abs(z_gas_at_R).max(),first)

      if (radius < 2.0 * dens_soft): N_back_in_bin = np.int(0.3* N_in_bin)
      else:  N_back_in_bin = np.int(0.05* N_in_bin)

      #a dense envelope of cells surrounding the disk
      R_back_at_R = np.repeat(radius,N_back_in_bin)
      z_back_at_R = np.logspace(np.log10(first),np.log10(min(first*15.0,vertical_bins.max(),z_max)),N_back_in_bin)
      Delta_z_at_R = z_back_at_R.max() - (z_back_at_R[z_back_at_R < z_back_at_R.max()]).max()
      
      #above and below the disk
      R_back_at_R = np.append(R_back_at_R,R_back_at_R)
      z_back_at_R = np.append(z_back_at_R,-z_back_at_R)  
      
      #add some irregularity by randomly turning off some of these cells
      on = (rd.random_sample(R_back_at_R.shape[0]) <= 0.5)
      R_back_at_R = R_back_at_R[on]
      z_back_at_R = z_back_at_R[on]
      
      dens_back_at_R = local_density(radius,np.abs(z_back_at_R))#/1000
      press_back_at_R = local_pressure(radius,np.abs(z_back_at_R))#/1000
      mass_back_at_R = np.repeat(M_disk/N_gas/100.0,z_back_at_R.shape[0])

      dens_back_at_R[dens_back_at_R < DENSITY_CUTOFF] = DENSITY_CUTOFF
      press_back_at_R[dens_back_at_R < DENSITY_CUTOFF] = DENSITY_CUTOFF * SOUNDSPEED_SQUARED_FLOOR

      mass_back = np.append(mass_back,mass_back_at_R)
      dens_back = np.append(dens_back,dens_back_at_R)
      press_back = np.append(press_back,press_back_at_R) 
      z_back = np.append(z_back,z_back_at_R)
      R_back = np.append(R_back,R_back_at_R)

      Vel_Corr_back = np.append(Vel_Corr_back,
                                derivative(vecSoundSpeedSquared,R_back_at_R)*np.log(dens_back_at_R/rho_plane) * radius)

               
      #again, we add more mesh at a coarser rate above the previous one.
      if (k % 2 == 0): #only every other radial bin
        if (vertical_bins.max() < z_max):
          N_back_in_bin = N_back_in_bin/4 #reduce the number of sampled locations
          R_back_at_R = np.repeat(radius,N_back_in_bin)
          # to avoid mesh overlap, we start at the maximum sampled z-value already present
          beta = 1 - np.exp(-radius/R_disk)
          z_back_at_R = np.logspace(np.log10(np.abs(z_back_at_R).max()+ Delta_z_at_R),np.log10(beta*z_max+(1-beta)*np.abs(z_back_at_R).max()),N_back_in_bin-1)
          z_back_at_R = np.append(z_back_at_R,beta*z_max+(1-beta)*vertical_bins.max())

          #above and below the disk
          R_back_at_R = np.append(R_back_at_R,R_back_at_R)
          z_back_at_R = np.append(z_back_at_R,-z_back_at_R)  
          #add some irregularity by randomly turning off some of these cells
          on = (rd.random_sample(R_back_at_R.shape[0]) <= 0.5)
          R_back_at_R = R_back_at_R[on]
          z_back_at_R = z_back_at_R[on]
      
          if (z_back_at_R.shape[0] > 0):
            dens_back_at_R = local_density(radius,np.abs(z_back_at_R))#/1000
            press_back_at_R = local_pressure(radius,np.abs(z_back_at_R))#/1000
            mass_back_at_R = np.repeat(M_disk/N_gas/10000.0,z_back_at_R.shape[0])
          
            dens_back_at_R[dens_back_at_R < DENSITY_CUTOFF] = DENSITY_CUTOFF
            press_back_at_R[dens_back_at_R < DENSITY_CUTOFF] = DENSITY_CUTOFF * SOUNDSPEED_SQUARED_FLOOR

            mass_back = np.append(mass_back,mass_back_at_R)
            dens_back = np.append(dens_back,dens_back_at_R)
            press_back = np.append(press_back,press_back_at_R) 
            z_back = np.append(z_back,z_back_at_R)
            R_back = np.append(R_back,R_back_at_R)

            Vel_Corr_back = np.append(Vel_Corr_back,np.zeros(R_back_at_R.shape[0]))

            # we want to save for later how far in the vertical direction we have accurately mapped the disk volume density
            max_z_back = np.append(max_z_back,beta*z_max+(1-beta)*vertical_bins.max())
            R_back_bins = np.append(R_back_bins,R_back_at_R.mean())
        else:
          if (z_back_at_R.shape[0] > 0):
            max_z_back = np.append(max_z_back,vertical_bins.max())
            R_back_bins = np.append(R_back_bins,R_back_at_R.mean())         

  if(SelfGravity):
    
    #radial potential gradient
    vcircsquared_0 = G * GasMass(radius)/radius 
    delta_vcirc,delta_vcirc_old  = 0.0, 1.0e20
    k1 = 1
    if (count >0):
      while(True):
        def integrand1(x): return GasSigma(x * R_disk) * x**(2.0*k1+1)
        def integrand2(x): return GasSigma(x * R_disk) / x**(2.0*k1)
        alpha_k1 = np.pi*(factorial(2*k1)/2.0**(2*k1)/factorial(k1)**2)**2
        
        integral1 = quad(integrand1,0.0,radius/R_disk,points=np.linspace(0.0,radius/R_disk,10))[0]
        integral2 = quad(integrand2,radius/R_disk,np.inf,limit=100)[0]
        
        delta_vcirc+=2.0 * alpha_k1 * G * R_disk *((2*k1+1.0)/(radius/R_disk)**(2*k1+1)* integral1 - 2.0*k1 *(radius/R_disk)**(2*k1) *integral2)

        if (k1 > 30):break
        abstol,reltol = 1.0e-6,1.0e-5
        abserr,relerr = np.abs(delta_vcirc_old-delta_vcirc),np.abs(delta_vcirc_old-delta_vcirc)/np.abs(delta_vcirc_old+vcircsquared_0)
        if (np.abs(delta_vcirc) > abstol/reltol):
          if (abserr < abstol): break
        else:
          if (relerr < reltol): break

        delta_vcirc_old = delta_vcirc
        k1 = k1 + 1
     
      selfgravity_vcirc_in_plane = np.append(selfgravity_vcirc_in_plane,vcircsquared_0+delta_vcirc)

  else:

    selfgravity_vcirc_in_plane = np.append(selfgravity_vcirc_in_plane,0.0)


################
print "   Maximum number of iterations:",max_vert_iter  

#numerically computed scale height
vecGasScaleHeightNumeric = InterpolatedUnivariateSpline(np.append(-radius_per_bin[0],radius_per_bin),\
                                                        np.append(local_scale_height[0],local_scale_height),k=3)
vecEffectiveGasSigma = InterpolatedUnivariateSpline(np.append(0.0,radius_per_bin),
                                                    np.append(SURFACE_DENSITY_FLOOR,effective_gas_sigma),
                                                    k=1)

plot_effective_sigma=1
if (plot_effective_sigma):
  plt.plot(np.linspace(R_min,R_cut,1000),vecEffectiveGasSigma(np.linspace(R_min,R_cut,1000)))
  plt.plot(radius_per_bin,effective_gas_sigma,'r+')
  plt.show()
def GasMass_Integrand(R): return vecEffectiveGasSigma(R) * R * 2 * np.pi
print "MASS", quad(GasMass_Integrand,0.0,R_cut,limit=70)[0]

if (output_scaleheight):
  f = open('scaleheight_'+ic_filename+'.txt','w')
  f.writelines( [ "%12.7f %12.7f \n" %(radius_per_bin[i], local_scale_height[i]) for i in range(0,radius_per_bin.shape[0])])
  f.close





#eliminate extreme outliers from the sample
print R_gas.shape
ind = (np.abs(z_gas) < vecGasScaleHeightNumeric(R_gas)*scale_height_cut) & (np.abs(z_gas) < z_max)
R_gas, z_gas = R_gas[ind],z_gas[ind]
dens_gas, press_gas= dens_gas[ind], press_gas[ind]
Vel_Correction = Vel_Correction[ind]

vecRhoGasInPlane = InterpolatedUnivariateSpline(np.append(-radius_per_bin[0],radius_per_bin),\
                                                np.append(density_in_plane[0],density_in_plane),k=3)
vecPressGasInPlane = InterpolatedUnivariateSpline(np.append(-radius_per_bin[0],radius_per_bin),\
                                                  np.append(pressure_in_plane[0],pressure_in_plane),k=3)
vecSelfGravityVcircInPlane = InterpolatedUnivariateSpline(np.append(0.0,radius_per_bin),\
                                                    np.append(selfgravity_vcirc_in_plane[0],selfgravity_vcirc_in_plane),k=3)

Pressure_Gradient = vecDerivative(vecPressGasInPlane,R_gas)


if (add_background): vecMaxZBack = InterpolatedUnivariateSpline(R_back_bins,max_z_back,k=3)

plt.plot(R_back_bins,max_z_back,'b.')
plt.plot(R_back_bins,vecMaxZBack(R_back_bins),'r')
plt.show()

#we truncate the pressure contours
print "\ntruncating..."
DENSITY_EDGE = vecRhoGasInPlane(R_cut)


DENSITY_CUT = vecRhoGasInPlane(R_cut)
DENSITY_BACKGROUND = DENSITY_EDGE/100.0
if (DENSITY_BACKGROUND < dens_back.min()): 
  DENSITY_BACKGROUND = dens_back.min()

PRESSURE_BACKGROUND = DENSITY_BACKGROUND * SOUNDSPEED_SQUARED_FLOOR * GAMMA_PRIME
#if (add_background) & (not load_mesh):
  #press_back[dens_back < DENSITY_BACKGROUND] = PRESSURE_BACKGROUND
  #dens_back[dens_back < DENSITY_BACKGROUND] = DENSITY_BACKGROUND
  
ind = (R_gas > R_cut) | (press_gas <= PRESSURE_BACKGROUND) | (np.abs(z_gas) > vecGasScaleHeightNumeric(R_gas)*scale_height_cut)
if (DENSITY_BACKGROUND < DENSITY_FLOOR): DENSITY_BACKGROUND = DENSITY_FLOOR
press_gas[ind] = PRESSURE_BACKGROUND
dens_gas[ind]= DENSITY_BACKGROUND
Pressure_Gradient[ind]=0.0
Vel_Correction[ind] = 0.0
dens_gas[dens_gas < 0] = DENSITY_BACKGROUND

#repeat for the surrounding background mesh
if(add_background):
  ind = (press_back <= PRESSURE_BACKGROUND) | (np.abs(z_back) > vecGasScaleHeightNumeric(R_back)*scale_height_cut)
  press_back[ind] = PRESSURE_BACKGROUND
  dens_back[ind]= DENSITY_BACKGROUND
  dens_back[dens_back < 0] = DENSITY_BACKGROUND


Real_M_disk = GasMass(R_cut)
gas_mass = Real_M_disk / N_gas

print "\ndone."

#fixing the center
print "Regularizing center..."
inner_cell = 0.6*MasstoRadius(gas_mass)
rad_gas = np.sqrt(R_gas**2 + z_gas**2)
ind = (rad_gas > inner_cell)
print "   removing ",ind[ind == False].shape[0]," central cells within R=",inner_cell
R_gas,z_gas = R_gas[ind],z_gas[ind]
dens_gas, press_gas, Pressure_Gradient = dens_gas[ind], press_gas[ind], Pressure_Gradient[ind]
Vel_Correction = Vel_Correction[ind]
#add a central cell
central_rho = max(vecRhoGasInPlane(0.0),DENSITY_BACKGROUND)
central_press = GasSoundSpeedSquared(0.0)*central_rho
R_gas = np.append(R_gas,rd.random_sample(1)[0]*R_min/100.0)
z_gas =np.append(z_gas,rd.random_sample(1)[0]*R_min/100.0)
dens_gas=np.append(dens_gas,central_rho)
press_gas=np.append(press_gas,central_press)
Pressure_Gradient = np.append(Pressure_Gradient,0.0)
Vel_Correction = np.append(Vel_Correction,0.0)
print "   adding central cells of density and pressure", central_rho,central_press
N_gas = R_gas.shape[0]
print "Total number of mesh-generating points in the model=",N_gas
N_in_disk = dens_gas[dens_gas > DENSITY_BACKGROUND].shape[0]
ids = np.zeros(N_gas)
ids[dens_gas > DENSITY_BACKGROUND] = np.arange(1,N_in_disk+1)
ids[dens_gas <= DENSITY_BACKGROUND] = np.arange(N_in_disk+1,N_gas + 1)

print "done."



#setting up cartesian positions
phi_gas=2.0*np.pi*rd.random_sample(N_gas)
x_gas=R_gas*np.cos(phi_gas)
y_gas=R_gas*np.sin(phi_gas)


#ind= (R_gas<R_cut)
#N_gas=R_gas[ind].shape[0]
#R_gas=R_gas[ind]
#x_gas=x_gas[ind]
#y_gas=y_gas[ind]
#z_gas=z_gas[ind]
#phi_gas=phi_gas[ind]
#dens_gas = dens_gas[ind]
#press_gas = press_gas[ind]

print "Sampling velocity structure..."

VelocityR_gas=np.zeros(N_gas)
VelocityPhi_gas=np.zeros(N_gas)
VelocityZ_gas=np.zeros(N_gas)

VelocityPhi_gasSquared= vecOmegaKeplerSquared(R_gas,0.0)* R_gas*R_gas + vecSelfGravityVcircInPlane(R_gas) + \
                        Pressure_Gradient/vecRhoGasInPlane(R_gas) * R_gas - Vel_Correction 

VelocityPhi_gasSquared[VelocityPhi_gasSquared < 0.0] = 0.0
VelocityPhi_gasSquared[(R_gas > R_cut)] = 0.0
VelocityPhi_gas = np.sqrt(VelocityPhi_gasSquared)

#if (output_velocity_field):
#  f = open('velocity-field_'+ic_filename+'.txt','w')
#  f.writelines( [ "%12.7f %12.7f %12.7f %12.7f %12.7f %12.7f\n" %(R_gas[i],VelocityPhi_gasSquared[i],
#                                                                  vecOmegaKeplerSquared(R_gas[i],0.0)* R_gas[i]*R_gas[i],
#                                                                  vecSelfGravityVcircInPlane(R_gas[i]),
#                                                                  Pressure_Gradient[i]/vecRhoGasInPlane(R_gas[i]) * R_gas[i],
#                                                                  - Vel_Correction[i]) for i in range(0,N_gas)])
#  f.close
if (output_velocity_field):
  f = open('velocity-field_'+ic_filename+'.txt','w')
  f.writelines( [ "%12.7f %12.7f %12.7f %12.7f %12.7f %12.7f\n" %
                  (radius_per_bin[i],
                   vecOmegaKeplerSquared(radius_per_bin[i],0.0)* radius_per_bin[i]*radius_per_bin[i]+
                   vecSelfGravityVcircInPlane(radius_per_bin[i])+
                    vecDerivative(vecPressGasInPlane,radius_per_bin[i])/vecRhoGasInPlane(radius_per_bin[i]) * radius_per_bin[i],
                   vecOmegaKeplerSquared(radius_per_bin[i],0.0)* radius_per_bin[i]*radius_per_bin[i],
                   vecSelfGravityVcircInPlane(radius_per_bin[i]),
                    vecDerivative(vecPressGasInPlane,radius_per_bin[i])/vecRhoGasInPlane(radius_per_bin[i]) * radius_per_bin[i],
                   0.0)
                  for i in range(0,radius_per_bin.shape[0])])
  f.close


vx_gas=VelocityR_gas*np.cos(phi_gas)-VelocityPhi_gas*np.sin(phi_gas)
vy_gas=VelocityR_gas*np.sin(phi_gas)+VelocityPhi_gas*np.cos(phi_gas)
vz_gas=VelocityZ_gas

if (add_background) & (not load_mesh):
  VelocityPhi_backSquared = vecOmegaKeplerSquared(R_back,0.0)* R_back*R_back + vecSelfGravityVcircInPlane(R_back) + \
                            vecDerivative(vecPressGasInPlane,R_back)/vecRhoGasInPlane(R_back) * R_back - Vel_Corr_back
  VelocityPhi_backSquared[VelocityPhi_backSquared < 0.0] = 0.0
  VelocityPhi_back = np.sqrt(VelocityPhi_backSquared)



print "done."  


print "Gas mass and internal energy..."
mass = np.zeros(N_gas) + gas_mass
utherm=press_gas/dens_gas/(GAMMA - 1)
print "cut=", R_gas.max(), R_gas.max(), R_cut
print "done"


########


if (add_background):
  x_background, y_background , z_background , radius_background = \
                np.array([]) , np.array([]) ,np.array([]),np.array([])
  vx_background, vy_background , vz_background = \
                 np.array([]) , np.array([]) ,np.array([])
  dens_background,press_background = np.array([]),np.array([])

  print "Creating background grid..."

  if (not load_mesh):
    print "   Background stage #0 (dense envelope around disk)..."
    if (DENSITY_BACKGROUND < DENSITY_FLOOR): DENSITY_BACKGROUND = DENSITY_FLOOR
  
    #first, we include the previously calculated background/transition mesh
    phi_back = rd.random_sample(R_back.shape[0])*2.0*np.pi
    x_background=np.append(x_background,R_back*np.cos(phi_back))
    y_background=np.append(y_background,R_back*np.sin(phi_back))
    z_background=np.append(z_background,z_back)

    #dens_back[dens_back < DENSITY_BACKGROUND] = DENSITY_BACKGROUND
    #press_back[dens_back < DENSITY_BACKGROUND] = PRESSURE_BACKGROUND
    
    dens_background=np.append(dens_background,dens_back)
    press_background=np.append(press_background,press_back)
    #press_background=np.append(press_background,dens_back*SOUNDSPEED_SQUARED_FLOOR)
    
    #this immediately neighboring background mesh DOES rotate with the disk
    vx_background=np.append(vx_background,-VelocityPhi_back*np.sin(phi_back))
    vy_background=np.append(vy_background,VelocityPhi_back*np.cos(phi_back))
    vz_background=np.append(vz_background,np.zeros(R_back.shape[0]))
    
    print "      adding ",R_back.shape[0], "points..."
    print "               average density :",dens_back.mean()
    print "               minimum density :",dens_back.min()
    print "               average pressure :",press_back.mean()
    print "               minimum pressure :",press_back.min()

  print "   Background stage #1..."
 
  low_res_size = 0.4*(gas_mass/DENSITY_EDGE/np.pi/4.0*3.0)**0.33
  meanvolume = low_res_size**3
  if (low_res_size > R_disk): low_res_size = 0.5 * R_disk
  if (low_res_size < R_disk*0.05): low_res_size = 0.05 * R_disk


  print "      creating background cells of approximate size (%5.1f)^3" %low_res_size 
  print z_max, vecMaxZBack([R_back.max()])

  R_ref = max(R_cut*1.05,R_max,R_back.max())
  x, y, z = np.mgrid[(-BOX_WIDTH+low_res_size)/2:(BOX_WIDTH-low_res_size)/2: 2.0*low_res_size,
                     (-BOX_WIDTH+low_res_size)/2:(BOX_WIDTH-low_res_size)/2: 2.0*low_res_size,
                     -2*z_max: 2 * z_max : 2.0 *low_res_size]
  x_back, y_back, z_back = x.flatten(), y.flatten(), z.flatten()
  x, y, z = np.mgrid[(-BOX_WIDTH/2+1.5*low_res_size):(BOX_WIDTH/2-1.5*low_res_size): 2.0*low_res_size,
                     (-BOX_WIDTH/2+1.5*low_res_size):(BOX_WIDTH/2-1.5*low_res_size): 2.0 *low_res_size,
                     (-2*z_max + low_res_size):(2 * z_max - low_res_size): 2.0 *low_res_size]
  x_back, y_back, z_back = np.append(x_back,x.flatten()),np.append(y_back,y.flatten()),np.append(z_back,z.flatten())
  
  R_back=np.sqrt(x_back*x_back + y_back*y_back)
  print R_back.max(),R_back.min(),np.abs(z_back).max(),np.abs(z_back).min(),R_ref
  ind = ((R_back < 1.05* R_ref) & (np.abs(z_back) > vecMaxZBack(R_back)))
  #plt.loglog(radius_per_bin,vecMaxZBack(radius_per_bin),'ko',ms=0.3)
  #plt.show()
  print "      adding ",x_back.shape[0]
  x_back, y_back, z_back = x_back[ind], y_back[ind], z_back[ind]
  R_back = R_back[ind]
  
  dens_back = np.zeros(x_back.shape[0])+DENSITY_BACKGROUND
  press_back = np.zeros(x_back.shape[0])+DENSITY_BACKGROUND*SOUNDSPEED_SQUARED_FLOOR
  print "      adding ",x_back.shape[0], "points..."
  if (x_back.shape[0] >0):
    print "               average density :",dens_back.mean()
    print "               minimum density :",dens_back.min()
    print "               average pressure :",press_back.mean()
    print "               minimum pressure :",press_back.min()
    
  x_background=np.append(x_background,x_back)
  y_background=np.append(y_background,y_back)
  z_background=np.append(z_background,z_back)
  
  dens_background=np.append(dens_background,dens_back)
  press_background=np.append(press_background,press_back)

  vx_background=np.append(vx_background,np.zeros(R_back.shape[0]))
  vy_background=np.append(vy_background,np.zeros(R_back.shape[0]))
  vz_background=np.append(vz_background,np.zeros(R_back.shape[0]))
  
  if (BOX_HEIGHT/2 > z_background.max()):
    print "   Background stage #2..."
    DENSITY_BACKGROUND = DENSITY_FLOOR
    PRESSURE_BACKGROUND = DENSITY_FLOOR * TEMPERATURE_FLOOR * \
                          BOLTZMANN_CGS / meanweight / PROTON_MASS_CGS \
                          / UnitEnergy_in_cgs * UnitMass_in_g
    if (DENSITY_BACKGROUND < DENSITY_FLOOR): DENSITY_BACKGROUND = DENSITY_FLOOR
    R_ref = R_back.max()
    low_res_size = 2* (gas_mass/DENSITY_EDGE/np.pi/4.0*3.0)**0.33
    maxvolume = low_res_size**3
    print "      creating background cells of approximate size (%5.1f)^3" %low_res_size 

    x, y, z = np.mgrid[-0.5*BOX_WIDTH:0.5*BOX_WIDTH: 2.0*low_res_size,
                       -0.5*BOX_WIDTH:0.5*BOX_WIDTH: 2.0*low_res_size,
                       -4*z_max: 4 * z_max : 2.0 *low_res_size]
    x_back, y_back, z_back = x.flatten(), y.flatten(), z.flatten()
    x, y, z = np.mgrid[-0.5*BOX_WIDTH+low_res_size:0.5*BOX_WIDTH: 2.0*low_res_size,
                       -0.5*BOX_WIDTH+low_res_size:0.5*BOX_WIDTH: 2.0 *low_res_size,
                       (-4*z_max + low_res_size):4 * z_max: 2.0 *low_res_size]
    x_back, y_back, z_back = np.append(x_back,x.flatten()),np.append(y_back,y.flatten()),np.append(z_back,z.flatten())

    R_back=np.sqrt(x_back*x_back + y_back*y_back)
    ind = ((np.abs(z_back) > z_background.max()) & (R_back < 1.3 *R_ref )) \
           | ((R_back > R_ref) & (R_back < 1.3*R_ref ))
    x_back, y_back, z_back = x_back[ind], y_back[ind], z_back[ind]
    R_back = R_back[ind]

    dens_back = np.zeros(x_back.shape[0])+DENSITY_BACKGROUND
    press_back = np.zeros(x_back.shape[0])+PRESSURE_BACKGROUND
    
    print "      adding ",x_back.shape[0], "points..."
    if (x_back.shape[0] >0):
      print "               average density :",dens_back.mean()
      print "               minimum density :",dens_back.min()
      print "               average pressure :",press_back.mean()
      print "               minimum pressure :",press_back.min()

    
    x_background=np.append(x_background,x_back)
    y_background=np.append(y_background,y_back)
    z_background=np.append(z_background,z_back)

    dens_background=np.append(dens_background,dens_back)
    press_background=np.append(press_background,press_back)

    vx_background=np.append(vx_background,np.zeros(R_back.shape[0]))
    vy_background=np.append(vy_background,np.zeros(R_back.shape[0]))
    vz_background=np.append(vz_background,np.zeros(R_back.shape[0]))
    
  if ((BOX_HEIGHT/2 > z_background.max()) & (x_back.shape[0] >0)):
    print "   Background stage #3..."
    DENSITY_BACKGROUND = DENSITY_FLOOR
    PRESSURE_BACKGROUND = DENSITY_FLOOR * TEMPERATURE_FLOOR * \
                          BOLTZMANN_CGS / meanweight / PROTON_MASS_CGS \
                          / UnitEnergy_in_cgs * UnitMass_in_g
    R_ref = R_back.max()
    low_res_size = low_res_size * 2.5
    maxvolume = low_res_size**3
    print "      creating background cells of approximate size (%5.1f)^3" %low_res_size 

    x, y, z = np.mgrid[-0.5*BOX_WIDTH:0.5*BOX_WIDTH: 2.0*low_res_size,
                       -0.5*BOX_WIDTH:0.5*BOX_WIDTH: 2.0*low_res_size,
                       -0.5*BOX_HEIGHT:0.5*BOX_HEIGHT: 2.0 *low_res_size]
    x_back, y_back, z_back = x.flatten(), y.flatten(), z.flatten()
    x, y, z = np.mgrid[-0.5*BOX_WIDTH + low_res_size:0.5*BOX_WIDTH: 2.0*low_res_size,
                       -0.5*BOX_WIDTH + low_res_size:0.5*BOX_WIDTH: 2.0*low_res_size,
                       -0.5*BOX_HEIGHT+ low_res_size:0.5*BOX_HEIGHT: 2.0 *low_res_size]
    x_back, y_back, z_back = np.append(x_back,x.flatten()),np.append(y_back,y.flatten()),np.append(z_back,z.flatten())
    
    R_back=np.sqrt(x_back*x_back + y_back*y_back)
    ind = ((np.abs(z_back) > z_background.max()) & (np.abs(z_back)< 2.4 * z_background.max())\
           & (R_back < 1.8*R_ref)) \
           | ((R_back > R_ref) & (R_back < 1.8 * R_ref ) & (np.abs(z_back) < 2.4 *z_background.max()))
    x_back, y_back, z_back = x_back[ind], y_back[ind], z_back[ind]
    R_back = R_back[ind]

    dens_back = np.zeros(x_back.shape[0])+DENSITY_BACKGROUND
    press_back = np.zeros(x_back.shape[0])+PRESSURE_BACKGROUND

    print "      adding ",x_back.shape[0], "points..."
    if (x_back.shape[0] >0):
      print "               average density :",dens_back.mean()
      print "               minimum density :",dens_back.min()
      print "               average pressure :",press_back.mean()
      print "               minimum pressure :",press_back.min()
    
    x_background=np.append(x_background,x_back)
    y_background=np.append(y_background,y_back)
    z_background=np.append(z_background,z_back)
    
    dens_background=np.append(dens_background,dens_back)
    press_background=np.append(press_background,press_back)
    
    vx_background=np.append(vx_background,np.zeros(R_back.shape[0]))
    vy_background=np.append(vy_background,np.zeros(R_back.shape[0]))
    vz_background=np.append(vz_background,np.zeros(R_back.shape[0]))
    
  if ((BOX_HEIGHT/2 > z_background.max()) & (x_back.shape[0] >0)): 
    print "   Background stage #4..."
    DENSITY_BACKGROUND = DENSITY_FLOOR
    PRESSURE_BACKGROUND = DENSITY_FLOOR * TEMPERATURE_FLOOR * \
                          BOLTZMANN_CGS / meanweight / PROTON_MASS_CGS \
                          / UnitEnergy_in_cgs * UnitMass_in_g
    R_ref = R_back.max()
    low_res_size = low_res_size * 2.5
    maxvolume = low_res_size**3
    if (low_res_size > BOX_WIDTH/8.0): low_res_size = BOX_WIDTH/16.0
    print "      creating background cells of approximate size (%5.1f)^3" %low_res_size 

    x, y, z = np.mgrid[(-BOX_WIDTH+low_res_size)/2:BOX_WIDTH/2: 2.0*low_res_size,
                       (-BOX_WIDTH+low_res_size)/2:BOX_WIDTH/2: 2.0*low_res_size,
                       (-BOX_HEIGHT+low_res_size)/2:BOX_HEIGHT/2: 2.0 *low_res_size]
    x_back, y_back, z_back = x.flatten(), y.flatten(), z.flatten()
    x, y, z = np.mgrid[(-BOX_WIDTH/2+1.5*low_res_size):BOX_WIDTH/2: 2.0*low_res_size,
                       (-BOX_WIDTH/2+1.5*low_res_size):BOX_WIDTH/2: 2.0 *low_res_size,
                       (-BOX_HEIGHT/2+1.5*low_res_size):BOX_HEIGHT/2: 2.0 *low_res_size]
    x_back, y_back, z_back = np.append(x_back,x.flatten()),np.append(y_back,y.flatten()),np.append(z_back,z.flatten())

    R_back=np.sqrt(x_back*x_back + y_back*y_back)
    ind = ((np.abs(z_back) > z_background.max()) & (np.abs(z_back) < BOX_WIDTH/2)) \
           | (R_back > R_ref)
    x_back, y_back, z_back = x_back[ind], y_back[ind], z_back[ind]
    R_back = R_back[ind]

    dens_back = np.zeros(x_back.shape[0])+DENSITY_BACKGROUND
    press_back = np.zeros(x_back.shape[0])+PRESSURE_BACKGROUND

    print "      adding ",x_back.shape[0], "points..."
    if (x_back.shape[0] >0):
      print "               average density :",dens_back.mean()
      print "               minimum density :",dens_back.min()
      print "               average pressure :",press_back.mean()
      print "               minimum pressure :",press_back.min()
    
    x_background=np.append(x_background,x_back)
    y_background=np.append(y_background,y_back)
    z_background=np.append(z_background,z_back)

    dens_background=np.append(dens_background,dens_back)
    press_background=np.append(press_background,press_back)

    vx_background=np.append(vx_background,np.zeros(R_back.shape[0]))
    vy_background=np.append(vy_background,np.zeros(R_back.shape[0]))
    vz_background=np.append(vz_background,np.zeros(R_back.shape[0]))
       
 
  
  ind=np.abs(z_background)< BOX_HEIGHT/2


  x_gas = np.append(x_gas,x_background[ind]) + BOX_WIDTH * 0.5
  y_gas = np.append(y_gas,y_background[ind]) + BOX_WIDTH * 0.5
  z_gas = np.append(z_gas,z_background[ind]) + BOX_HEIGHT * 0.5

  N_BACKGROUND = len(x_background[ind])
  
  vx_gas = np.append(vx_gas,vx_background)
  vy_gas = np.append(vy_gas,vy_background)
  vz_gas = np.append(vz_gas,vy_background)
  mass = np.append(mass,np.repeat(gas_mass*1e-10,N_BACKGROUND))
  dens_gas = np.append(dens_gas,dens_background)
  utherm = np.append(utherm,press_background/dens_background/(GAMMA-1))

  UTHERM_BACKGROUND = (press_background/dens_background/(GAMMA-1)).min()
  dens_gas[dens_gas < DENSITY_FLOOR] = DENSITY_FLOOR
  
  N_total = N_gas + N_BACKGROUND
  ids = np.append(ids,np.arange(N_gas+1,N_total+1))

  print "done"
  
  N_gas = N_total

##########################################################
  
print "Writing snapshot..."
f=ws.openfile(ic_filename+".hdf5")

ws.write_block(f, "POS ", 0, np.array([x_gas,y_gas,z_gas]).T)
ws.write_block(f, "VEL ", 0, np.array([vx_gas,vy_gas,vz_gas]).T)
ws.write_block(f, "U   ", 0, utherm)
if (relax_density):
  ws.write_block(f, "MASS", 0, dens_gas)
else:
  ws.write_block(f, "MASS", 0, mass)
ws.write_block(f, "ID  ", 0, ids)

if not (ExternalPotential):
  print "Adding central star..."
  if(add_background):
    ws.write_block(f, "POS ", 4, np.array([BOX_WIDTH * 0.5, BOX_WIDTH * 0.5, BOX_HEIGHT * 0.5]).T)
  else:
    ws.write_block(f, "POS ", 4,np.array([0.0,0.0,0.0]).T)
  ws.write_block(f, "VEL ", 4, np.array([0.0,0.0,0.0]).T)
  ws.write_block(f, "MASS", 4, np.array([M_star]))
  ws.write_block(f, "ID  ", 4, np.array([N_gas + 1]))
    
  npart=np.array([N_gas,0,0,0,1,0], dtype="uint32")
else:
  npart=np.array([N_gas,0,0,0,0,0], dtype="uint32")

massarr=np.array([0,0,0,0,0,0], dtype="float64")
header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr, double = np.array([double_precision], dtype="int32"))
ws.writeheader(f, header)
ws.closefile(f)
print "done."

print x_gas.shape[0],mass.shape,dens_gas.shape,vx_gas.shape[0],utherm.shape

print "\n-FINISHED-\n"

if not (add_background):
  print "N_gas  = ", N_gas
if (add_background):
  print "N_gas(in disk)  = ", N_in_disk
  print "N(background grid)  = ", N_BACKGROUND

print "M_disk=",Real_M_disk,GasMass(R_max), GasMass(R_cut)

print "DENSITY EDGE=",DENSITY_EDGE
print "DENSITY CUT=",DENSITY_CUT
print "DENSITY CUTOFF=",DENSITY_CUTOFF
print "DENSITY FLOOR=",DENSITY_FLOOR
print "SOUNDSPEED_SQUARED_FLOOR=",SOUNDSPEED_SQUARED_FLOOR
if (add_background):
  print "DENSITY BACKGROUND=",DENSITY_BACKGROUND
  print "PRESSURE BACKGROUND=",PRESSURE_BACKGROUND
  print "UTHERM BACKGROUND=",UTHERM_BACKGROUND
print "MEAN CELL VOLUME (in disk)=",(1.0/(dens_gas[dens_gas > DENSITY_EDGE])).max()*gas_mass  

print "\nFOR THE PARAMETER FILE #####################################\n"

print "ReferenceGasPartMass","%5.3e" % gas_mass
print "IrradiationTempScaling",GasTemperature(0.0) / np.sqrt(M_star) / SOLAR_EFF_TEMP
print "MinVolume", "%5.3e" % (gas_mass/dens_gas.max())
if ((meanvolume > 1) & (meanvolume <=10)):
  print "MaxVolume","%4.2e" % (np.ceil(meanvolume))
  print "MeanVolume","%4.2e" % (np.ceil(meanvolume))
if ((meanvolume > 10) & (meanvolume <=100)):
  print "MaxVolume","%4.2e" % (np.ceil(maxvolume * 0.1) * 10)
  print "MeanVolume","%4.2e" % (np.ceil(meanvolume * 0.1) * 10)
if ((meanvolume > 100) & (meanvolume <=1000)):
  print "MaxVolume","%4.2e" % (np.ceil(maxvolume * 0.01) * 100)
  print "MeanVolume","%4.2e" % (np.ceil(meanvolume * 0.01) * 100)
if (meanvolume > 1000):
  print "MeanVolume","%4.2e" % (np.ceil(meanvolume * 0.001) * 1000)
  print "MaxVolume","%4.2e" % (np.ceil(maxvolume * 0.001) * 1000)
print "SofteningStars",(star_soft/2.8)
print "SofteningGas","%3.1e" % (3*R_min/2.8) 
print "BoxSize",BOX_HEIGHT
print "MinGasTemp    0.0"
print "MinEgySpec",UTHERM_FLOOR
if (add_background):
  print "MinimumDensityOnStartUp",DENSITY_FLOOR
  print "LimitUBelowThisDensity", DENSITY_EDGE/100.0
  print "LimitUBelowCertainDensityToThisValue",UTHERM_BACKGROUND
 
