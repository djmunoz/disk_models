import numpy as np



def powerlaw_sigma(R,sigma0,p,R0):
    return sigma0*(R/R0)**(-p)

def similarity_sigma(R,sigma0,gamma,Rc):
    return sigma0*(R/Rc)**(-gamma) * np.exp(-(R/Rc)**(2.0-gamma))

def similarity_softened_sigma(R,sigma0,gamma,Rc,soft_length):
    return sigma0*(SplineProfile(R,soft_length) * Rc)**(gamma) * np.exp(-(R/Rc)**(2.0-gamma))

def similarity_hole_sigma(R,sigma0,gamma,Rc,R_hole):
    return sigma0*(SplineDerivative(R,R_hole) * R  * R * Rc)**(gamma) * np.exp(-(R/Rc)**(2.0-gamma))


def powerlaw_cavity_sigma(R,sigma0,p,xi,R_cav):
    return sigma0 * (R_cav/R)**p * np.exp(-(R_cav/R)**xi) 

def similarity_cavity_sigma(R,sigma0,gamma,Rc,xi,R_cav):
    return sigma0*(R/Rc)**(-gamma) * np.exp(-(R/Rc)**(2.0-gamma)) * np.exp(-(R_cav/R)**xi) 


def SplineProfile(R,h):
    r2 = R*R
    h_inv = 1.0 / h
    h3_inv = h_inv * h_inv * h_inv
    u = R * h_inv
    wp = -1.0 / R
    
    if (isinstance(R,np.ndarray)):
        type(R),type(u),type(wp)
        ind = u < 0.5
        wp[ind] = h_inv * (-2.8 + u[ind] * u[ind] * (5.333333333333 + u[ind] * u[ind] * (6.4 * u[ind] - 9.6)))
        ind = (u >= 0.5) & (u < 1)
        wp[ind] = h_inv * (-3.2 + 0.066666666667 / u[ind] + u[ind] * u[ind] * (10.666666666667 + u[ind] * (-16.0 + u[ind] * (9.6 - 2.133333333333 * u[ind]))))
    else:
        if(R < h):
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
  h_inv = 1.0 / h
  h3_inv = h_inv * h_inv * h_inv
  u = R * h_inv
  fac = 1.0 / (r2 * R)

  if (isinstance(R,(list, tuple, np.ndarray))):
      ind = u < 0.5
      fac[ind] = h3_inv * (10.666666666667 + u[ind] * u[ind] * (32.0 * u[ind] - 38.4))
      ind = (u >= 0.5) & (u < 1)
      fac[ind] = h3_inv * (21.333333333333 - 48.0 * u[ind] + 38.4 * u[ind] * u[ind] - 10.666666666667 * u[ind] * u[ind] * u[ind] - 0.066666666667 / \
                           (u[ind] * u[ind] * u[ind]))
  else:
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
        self.floor = kwargs.get("floor") 
        
        #set default values
        if (self.sigma0 is None):
            self.sigma0 = 1.0
        if (self.p is None):
            self.p = 1.0
        if (self.R0 is None):
            self.R0 = 1.0
        if (self.floor is None):
            self.floor = 0.0

    def evaluate(self,R):
        return np.maximum(self.floor,powerlaw_sigma(R,self.sigma0,self.p,self.R0))

class similarity_disk(object):
    def __init__(self, *args, **kwargs):
        self.sigma0 = kwargs.get("sigma0")
        self.gamma = kwargs.get("gamma")
        self.Rc = kwargs.get("Rc")
        self.floor = kwargs.get("floor") 
        
        #set default values
        if (self.sigma0 is None):
            self.sigma0 = 1.0
        if (self.gamma is None):
            self.gamma = 1.0
        if (self.Rc is None):
            self.Rc = 1.0
        if (self.floor is None):
            self.floor = 0.0

        print self.Rc,self.gamma,self.sigma0

    def evaluate(self,R):
        return np.maximum(self.floor,similarity_sigma(R,self.sigma0,self.gamma,self.Rc))

class similarity_softened_disk(object):
    def __init__(self, *args, **kwargs):
        self.sigma0 = kwargs.get("sigma0")
        self.gamma = kwargs.get("gamma")
        self.Rc = kwargs.get("Rc")
        self.floor = kwargs.get("floor")
        self.sigma_soft = kwargs.get("sigma_soft")
        
        #set default values
        if (self.sigma0 is None):
            self.sigma0 = 1.0
        if (self.gamma is None):
            self.gamma = 1.0
        if (self.Rc is None):
            self.Rc = 1.0
        if (self.floor is None):
            self.floor = 0.0
        if (self.sigma_soft is None):
            self.sigma_soft = 1e-5

    def evaluate(self,R):
        return np.maximum(self.floor,similarity_softened_sigma(R,self.sigma0,self.gamma,self.Rc,self.sigma_soft))


class similarity_hole_disk(object):
    def __init__(self, *args, **kwargs):
        self.sigma0 = kwargs.get("sigma0")
        self.gamma = kwargs.get("gamma")
        self.Rc = kwargs.get("Rc")
        self.floor = kwargs.get("floor")
        self.R_hole = kwargs.get("R_hole")
        
        #set default values
        if (self.sigma0 is None):
            self.sigma0 = 1.0
        if (self.gamma is None):
            self.gamma = 1.0
        if (self.Rc is None):
            self.Rc = 1.0
        if (self.floor is None):
            self.floor = 0.0
        if (self.R_hole is None):
            self.R_hole = 1e-5

    def evaluate(self,R):
        return np.maximum(self.floor,similarity_hole_sigma(R,self.sigma0,self.gamma,self.Rc,self.R_hole))
    
class powerlaw_cavity_disk(object):
    def __init__(self, *args, **kwargs):

        self.sigma0 = kwargs.get("sigma0")
        self.p = kwargs.get("p")
        self.R_cav = kwargs.get("R_cav")
        self.xi = kwargs.get("xi")
        self.floor = kwargs.get("floor") 

        #set default values
        if (self.sigma0 is None):
            self.sigma0 = 1.0
        if (self.p is None):
            self.p = 1.0
        if (self.R_cav is None):
            self.R_cav = 5.0
        if (self.xi is None):
            self.xi = 4.0
        if (self.floor is None):
            self.floor = 0.0
            
    def evaluate(self,R):
        return np.maximum(self.floor,powerlaw_cavity_sigma(R,self.sigma0,self.p,self.R_cav,self.xi))


class similarity_cavity_disk(object):
    def __init__(self, *args, **kwargs):

        self.sigma0 = kwargs.get("sigma0")
        self.gamma = kwargs.get("gamma")
        self.Rc = kwargs.get("Rc")
        self.R_cav = kwargs.get("R_cav")
        self.xi = kwargs.get("xi")
        self.floor = kwargs.get("floor") 
        
        #set default values
        if (self.sigma0 is None):
            self.sigma0 = 1.0
        if (self.gamma is None):
            self.gamma = 1.0
        if (self.Rc is None):
            self.Rc = 1.0
        if (self.R_cav is None):
            self.R_cav = 1.0
        if (self.xi is None):
            self.xi = 4.0
        if (self.floor is None):
            self.floor = 0.0

    def evaluate(self,R):
        return np.maximum(self.floor,similarity_cavity_sigma(R,self.sigma0,self.gamma,self.Rc,self.R_cav,self.xi))
