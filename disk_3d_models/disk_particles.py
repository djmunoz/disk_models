import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd

G = 1

def elements(x,y,z,vx,vy,vz,mu):

  R2 = x**2 + y**2 + z**2
  R = np.sqrt(R2)
  V2 = vx**2 + vy**2 + vz**2
  RtimesRdot = x * vx + y * vy + z * vz
  hx = y * vz - z * vy
  hy = z * vx - x * vz
  hz = x * vy - y * vx
  h2 = hx**2 + hy**2 + hz**2
  if (RtimesRdot > 0 ):
    Rdot = np.sqrt(V2 - h2/R2) 
  else:
    Rdot = -np.sqrt(V2 - h2/R2) 

  #eccentricity and pericenter distance
  mu_1 = 1.0/mu
  temp = 1.0  +  h2 * mu_1 * (V2 *mu_1  -  2.0 / R)
  if (temp <= 0): ecc = 0.0
  else: ecc = np.sqrt(temp)

  if (ecc < 1e-8): ecc = 1.e-8

  peridist = h2 * mu_1 / (1.0 + ecc)
  semimaj = 1.0 / (2.0/R - V2 * mu_1)

  #inclination
  incl = np.arccos(hz/np.sqrt(h2))
  if (incl != 0.0):
    if (hz > 0): node = np.arctan2(hx /np.sqrt(h2)/np.sin(incl),
                                   -hy/np.sqrt(h2)/np.sin(incl))
    else: node = np.arctan2(-hx /np.sqrt(h2)/np.sin(incl),
                            hy/np.sqrt(h2)/np.sin(incl))
  else:
    node = 0.0

  #true longitude (argument of pericenter plus true anomaly)   
  if ((incl > 1.e-3) & (incl < np.pi-1.0e-3)):
    sinomegaplusf = z/R/np.sin(incl)
    cosomegaplusf = 1.0/np.cos(node)*(x/R + np.sin(node) * sinomegaplusf *
                                      np.cos(incl))
    
    periargplustrue_anom = np.arctan2(sinomegaplusf,cosomegaplusf)
  else:
    periargplustrue_anom = np.arctan2(y,x)*np.cos(incl)


  #true anomaly and argument of pericenter
  true_anom = np.arctan2(semimaj*(1.0 -ecc**2)/np.sqrt(h2)/ecc * Rdot,
                         1.0/ecc*(semimaj/R*(1.0 - ecc**2) - 1.0))
  periarg = periargplustrue_anom - true_anom
    

  periarg = periarg % (2*np.pi)
  true_anom = true_anom % (2*np.pi)
  if (true_anom < 0): true_anom = 2.0*np.pi + true_anom
  if (periarg < 0): periarg = 2.0*np.pi + periarg


  if (ecc < 1.0):
    tanecc_anom_2 = np.tan(0.5 * true_anom)* np.sqrt((1.0 - ecc)/(1.0 + ecc))
    tanecc_anom = 2.0 * tanecc_anom_2 / (1.0 - tanecc_anom_2**2)
    cosecc_anom = (1.0 - R / semimaj )/ ecc
    ecc_anom = np.arctan2(tanecc_anom * cosecc_anom,cosecc_anom)
    if (ecc_anom < 0): ecc_anom = 2.0*np.pi + ecc_anom
    
    mean_anom = ecc_anom - ecc * np.sin(ecc_anom)
    
    return peridist,ecc,incl,periarg,node,true_anom,mean_anom,ecc_anom
      
  else:
    tanhhyp_anom_2 = np.tan(0.5 * true_anom)* np.sqrt((ecc - 1.0)/(ecc + 1.0))
    tanhhyp_anom = 2.0 * tanhhyp_anom_2 / (tanhhyp_anom_2**2 + 1.0)
    hyp_anom = np.arctanh(tanhhyp_anom)
    
    mean_anom = ecc * np.sinh(hyp_anom) - hyp_anom
    
    return peridist,ecc,incl,periarg,node,true_anom,mean_anom,hyp_anom


###################################################################################
def orbit(peri,e,I,omega,Omega,M,mu):

  
  
  if (e < 1):
    a = peri/(1.0 - e)
    ecc_anom = KeplerEquation(M,e)
    x = a * (np.cos(ecc_anom) - e)
    y = a * np.sqrt(1 - e * e) * np.sin(ecc_anom)
    xdot = -np.sqrt(mu/a) / (1.0 - e*np.cos(ecc_anom))* np.sin(ecc_anom)
    ydot = np.sqrt(mu/a) / (1.0 - e*np.cos(ecc_anom))* np.cos(ecc_anom) * np.sqrt(1.0 - e * e)
    f = np.arctan2(np.sqrt(1.0 - e * e) * np.sin(ecc_anom),np.cos(ecc_anom) - e)
    radius = a*(1.0 - e*np.cos(ecc_anom))
    
  elif (e > 1):
    a = peri/(1.0 - e)
    hyp_anom = HyperbolicKeplerEquation(M,e)
    x = a * (np.cosh(hyp_anom) - e)
    y = -a * np.sqrt(e * e - 1.0) * np.sinh(hyp_anom)
    xdot = -np.sqrt(mu/-a) / (e*np.cosh(hyp_anom)- 1.0)* np.sinh(hyp_anom)
    ydot = np.sqrt(mu/-a) / (e*np.cosh(hyp_anom) - 1.0)* np.cosh(hyp_anom) * np.sqrt(e * e - 1.0)
    f = np.arctan2(np.sqrt(e * e - 1.0) * np.sinh(hyp_anom),e - np.cosh(hyp_anom))
    radius = a*(1.0 - e * np.cosh(hyp_anom))
    
  elif (e == 1):
    tan_trueanom_2 = ParabolicKeplerEquation(M)
    x = peri * (1.0 - tan_trueanom_2**2)
    y = 2.0* peri * tan_trueanom_2
    xdot = -np.sqrt(2.0 * mu / peri) / (1.0 + tan_trueanom_2**2) * tan_trueanom_2
    ydot =  np.sqrt(2.0 * mu / peri) / (1.0 + tan_trueanom_2**2)
    f = np.arctan2(2.0 * tan_trueanom_2/(1.0 + tan_trueanom_2**2),(1.0 - tan_trueanom_2**2)/(1.0 + tan_trueanom_2**2))
    radius = peri * (1.0 + tan_trueanom_2**2)
        
    
  #fac = np.sqrt(mu/a/(1.0-e*e))

  
  #X = radius*(np.cos(Omega)*cos(omega+f) - \
  #                 np.sin(Omega)*sin(omega+f)*np.cos(I))
  #Y = radius*(np.sin(Omega)*cos(omega+f) + \
  #                 np.cos(Omega)*sin(omega+f)*np.cos(I))           
  #Z = radius*np.sin(omega+f)*np.sin(I)

  #VX = fac*(- np.sin(omega+f)- e*np.sin(omega))
  
  #VY = fac*np.cos(I)*(np.cos(omega+f) + e*np.cos(omega))
  
  #VZ = fac*np.sin(I)*(-np.cos(omega+f) - e*np.cos(omega))

  #rotation matrix
  d11 =  np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(I)
  d12 =  np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(I)
  d13 =  np.sin(omega) * np.sin(I)
  d21 = -np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.sin(Omega) * np.cos(I)
  d22 = -np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(Omega) * np.cos(I)
  d23 =  np.cos(omega) * np.sin(I)

  X = d11 * x + d21 * y
  Y = d12 * x + d22 * y
  Z = d13 * x + d23 * y
  VX = d11 * xdot + d21 * ydot
  VY = d12 * xdot + d22 * ydot
  VZ = d13 * xdot + d23 * ydot
      
  return X, Y, Z, VX, VY, VZ

###################################################################################3
def KeplerEquation(mean_anom,e):
  
  while (np.abs(mean_anom) > 2.0*np.pi):
    mean_anom-=2.0*np.pi*np.sign(mean_anom)
  if (mean_anom < 0.0): mean_anom = 2*np.pi + mean_anom

  k = 0.85
  ecc_anom = mean_anom + np.sign(np.sin(mean_anom))* k * e
  #ecc_anom = mean_anom
  if (e > 0.8):
    ecc_anom = np.pi
    
  abstol,reltol = 1.0e-8, 1.0e-8
  iter = 0
  while(True):
    f = ecc_anom -e * np.sin(ecc_anom) - mean_anom
    fprime = 1.0 - e * np.cos(ecc_anom)
    fprime2 = e * np.sin(ecc_anom)
    fprime3 = e * np.cos(ecc_anom)
    delta1 = - f / fprime
    delta2 = - f /(fprime + 0.5 * delta1 * fprime2)
    delta3 = - f /(fprime + 0.5 * delta2 * fprime2 + 0.16666666666 * delta2**2 * fprime3) 

    if (delta3 == 0): break
    
    if (np.abs(ecc_anom) > 0.0):
      abserr,relerr = np.abs(delta3),np.abs(delta3)/np.abs(ecc_anom)
    else:
      abserr,relerr = np.abs(delta3),1.0e40

    ecc_anom+=delta3
    #print iter,ecc_anom,e,delta3
    
    if (np.abs(ecc_anom) > abstol/reltol):
      if (abserr < abstol): break
    else:
      if (relerr < reltol): break
    iter+=1      

  return ecc_anom % (2*np.pi)

###################################################################################3
def HyperbolicKeplerEquation(mean_anom,e):

  #while (np.abs(mean_anom) > 2.0*np.pi):
  #  mean_anom-=2.0*np.pi*np.sign(mean_anom)
  #if (np.abs(mean_anom) > np.pi): mean_anom-=2.0*np.pi*np.sign(mean_anom)

  if (mean_anom > 0):
    hyp_anom = np.log(mean_anom/np.e + 1.8)
  else:
    hyp_anom = -np.log(np.abs(mean_anom)/np.e + 1.8)
  abstol,reltol = 1.0e-10, 1.0e-10
  iter = 0
  while(True):
    f = e * np.sinh(hyp_anom) - hyp_anom - mean_anom
    fprime =  e * np.cosh(hyp_anom) - 1.0
    fprime2 = e * np.sinh(hyp_anom) 
    fprime3 = e * np.cosh(hyp_anom)
    delta1 = - f / fprime
    delta2 = - f /(fprime + 0.5 * delta1 * fprime2)
    delta3 = - f /(fprime + 0.5 * delta2 * fprime2 + 0.16666666666 * delta2**2 * fprime3) 

    if (delta3 == 0): break
    
    if (np.abs(hyp_anom) > 0.0):
      abserr,relerr = np.abs(delta3),np.abs(delta3)/np.abs(hyp_anom)
    else:
      abserr,relerr = np.abs(delta3),1.0e40
      
    hyp_anom+=delta3
    #print iter,hyp_anom,e,delta3
    
    if (np.abs(hyp_anom) > abstol/reltol):
      if (abserr < abstol): break
    else:
      if (relerr < reltol): break
    iter+=1

      
  return hyp_anom


##################################################################
def ParabolicKeplerEquation(mean_anom): #jerome cardan's method

  #while (np.abs(mean_anom) > 2.0*np.pi):
  #  mean_anom-=2.0*np.pi*np.sign(mean_anom)
  #if (np.abs(mean_anom) > np.pi): mean_anom-=2.0*np.pi*np.sign(mean_anom)

  B = 1.5 * mean_anom
  tan_trueanom_2 = (B + np.sqrt(1 + B *  B))**(1.0/3) - 1.0/(B + np.sqrt(1 + B *  B))**(1.0/3)
  return tan_trueanom_2


##################################################################
def semimajor_axis(x,y,z,vx,vy,vz,mu):

    mu_1 = 1.0/mu
    R2 = x**2 + y**2 + z**2
    R = np.sqrt(R2)
    V2 = vx**2 + vy**2 + vz**2
    semimaj = 1.0 / (2.0/R - V2 * mu_1)
    
    return semimaj

def eccentricity(x,y,z,vx,vy,vz,mu):

    R2 = x**2 + y**2 + z**2
    R = np.sqrt(R2)
    V2 = vx**2 + vy**2 + vz**2
    RtimesRdot = x * vx + y * vy + z * vz
    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx
    h2 = hx**2 + hy**2 + hz**2
    if (RtimesRdot > 0 ):
        Rdot = np.sqrt(V2 - h2/R2) 
    else:
        Rdot = -np.sqrt(V2 - h2/R2) 
        
    mu_1 = 1.0/mu
    temp = 1.0  +  h2 * mu_1 * (V2 *mu_1  -  2.0 / R)
    if (temp <= 0): eccentricity = 0.0
    else: eccentricity = np.sqrt(temp)

    return eccentricity


##################################################################


class particle_data():
    def __init__(self,*args,**kwargs):
        self.pos=kwargs.get("pos")
        self.vel=kwargs.get("vel")
        self.mass=kwargs.get("mass")
        self.ids=kwargs.get("ids")

        if (self.pos is None):
            self.pos = np.empty([0,3])
        if (self.vel is None):
            self.vel = np.empty([0,3])
        if (self.mass is None):
            self.mass = np.empty([0])
        if (self.ids is None):
            self.ids = np.empty([0])

            
    def add_particle(self, x = 0.0, y = 0.0, z = 0.0, vx = 0.0, vy = 0.0, vz = 0.0, m = 0.0,
                     a = None, e = None, I = None, g = None, h = None, l = None, ID = None):

        if (self.pos.shape[0] > 0):
            print self.pos.shape
            mcm = self.mass.sum()
            poscm = self.pos.sum(axis = 0)/mcm
            velcm = self.vel.sum(axis = 0)/mcm
        
        if (a is not None):
            if (e is None):
                e = 1.0e-8
            if (I is None):
                I = 1.0e-10
            if (g is None):
                g = rd.random() * 2 * np.pi
            if (h is None):
                h = rd.random() * 2 * np.pi
            if (l is None):
                l = rd.random() * 2 * np.pi
            
            self.add_particle_orbital_elements(a, e, I, g, h, l, m,ID)

        else:
            self.pos = np.append(self.pos,np.array([x, y, z]).reshape(1,3), axis=0)
            self.vel = np.append(self.vel,np.array([vx, vy, vz]).reshape(1,3), axis=0)
            self.mass = np.append(self.mass,m)
            self.ids = np.append(self.ids,ID) 
            self.center_of_mass_frame()

    def add_particle_orbital_elements(self,a, e, I, g, h, l, m, ID):

        # In these cases, always assume the orbital elements are in Jacobian coordinates
        
        npart = self.pos.shape[0]
        mcm = self.mass.sum()
        poscm = np.array([(self.pos[:,0] * self.mass).sum()/mcm,
                          (self.pos[:,1] * self.mass).sum()/mcm,
                          (self.pos[:,2] * self.mass).sum()/mcm])
        velcm = np.array([(self.vel[:,0] * self.mass).sum()/mcm,
                          (self.vel[:,1] * self.mass).sum()/mcm,
                          (self.vel[:,2] * self.mass).sum()/mcm])


        distances = np.sqrt((self.pos[:,0]- poscm[0])**2+
                            (self.pos[:,1]- poscm[1])**2+
                            (self.pos[:,2]- poscm[2])**2)
        
        
        # check if the new particle is the most distant one
        if np.all(distances < a) | (len(self.pos) == 1):
            mu = G * (mcm + m)
            x,y,z,vx,vy,vz =  orbit(a*(1-e),e,I,g,h,l,mu)
            print self.pos
            self.pos = np.append(self.pos,(np.array([x, y, z])+poscm).reshape(1,3), axis=0)
            self.vel = np.append(self.vel,(np.array([vx, vy, vz])+velcm).reshape(1,3), axis=0)
            self.mass = np.append(self.mass,m)
            self.ids = np.append(self.ids,ID)
            print self.pos
            
        else: #if not, alter the existing particles
            ind_inner = np.argwhere(distances < a)
            ind_outer = np.argwhere(distances > a)

            for kk in ind_outer:
                menclosed =  self.mass[distances < distances[kk]].sum()
                posenclosed = self.pos[distances < distances[kk]].sum(axis=1)/menclosed
                # original orbital elements
                mu = G * (menclosed + self.mass[kk])                       
                peridist,ecc,incl,periarg,node,true_anom,_,_ = elements(self.pos[kk,0],self.pos[kk,1],self.pos[kk,2],
                                                                        self.vel[kk,0],self.vel[kk,1],self.vel[kk,2],
                                                                        mu) 
                # new cartesian coordinates
                mu = G * (menclosed + self.mass[kk] + m)                       
                x,y,z,vx,vy,vz =  orbit(peridist,ecc,incl,periarg,node,true_anom,mu)
                self.pos[kk] = np.array([x, y, z]).reshape(1,3),
                self.vel[kk] = np.array([vx, vy, vz]).reshape(1,3)

            # and for the newly added particle
            minner = self.mass[distances < a].sum()
            posinner = self.pos[distances < a].sum(axis = 1)/minner
            mu = G * (minner + m)
            x,y,z,vx,vy,vz =  orbit(a*(1-e),e,I,g,h,l,mu)
            self.pos = np.append(self.pos,np.array([x, y, z]).reshape(1,3), axis=0)
            self.vel = np.append(self.vel,np.array([vx, vy, vz]).reshape(1,3), axis=0)
            self.mass = np.append(self.mass,m)
            self.ids = np.append(self.ids,ID)

            # reorder the particles
            self.reorder_particles()

            # and move to the center-of-mass frame
            self.center_of_mass_frame()

            
        
    def center_of_mass_frame(self):
        npart = self.pos.shape[0]
        mcm = self.mass.sum()
        poscm = np.array([(self.pos[:,0] * self.mass).sum()/mcm,
                          (self.pos[:,1] * self.mass).sum()/mcm,
                          (self.pos[:,2] * self.mass).sum()/mcm])
        velcm = np.array([(self.vel[:,0] * self.mass).sum()/mcm,
                          (self.vel[:,1] * self.mass).sum()/mcm,
                          (self.vel[:,2] * self.mass).sum()/mcm])


        
        self.pos = np.subtract(self.pos,poscm)
        self.vel = np.subtract(self.vel,velcm)


    def reorder_particles(self):
        npart = self.pos.shape[0]
        mcm = self.mass.sum()
        poscm = np.array([(self.pos[:,0] * self.mass).sum()/mcm,
                          (self.pos[:,1] * self.mass).sum()/mcm,
                          (self.pos[:,2] * self.mass).sum()/mcm])


        
        distances = np.sqrt((self.pos[:,0]- poscm[0])**2+
                            (self.pos[:,1]- poscm[1])**2+
                            (self.pos[:,2]- poscm[2])**2)

        sorted_ind = np.argsort(distances)

        self.pos = self.pos[sorted_ind]
        self.vel = self.vel[sorted_ind]
        self.mass = self.mass[sorted_ind]
        self.ids = self.ids[sorted_ind]
    
        
