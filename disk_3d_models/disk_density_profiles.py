import numpy as np



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
        return max(self.floor,powerlaw_sigma(R,self.sigma0,self.p,self.R0))

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

    def evaluate(self,R):
        return max(self.floor,similarity_sigma(R,self.sigma0,self.gamma,self.Rc))

class powerlaw_cavity_disk(object):
    def __init__(self, *args, **kwargs):

        self.sigma0 = kwargs.get("sigma0")
        self.p = kwargs.get("p")
        self.R_cav = kwargs.get("R_cav")
        self.xi = kwargs.get("xi")


        #set default values
        if (self.sigma0 is None):
            self.sigma0 = 1.0
        if (self.p is None):
            self.p = 1.0
        if (self.R_cav is None):
            self.R_cav = 5.0
        if (self.xi is None):
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
        return max(self.floor,similarity_cavity_sigma(R,self.sigma0,self.gamma,self.Rc,self.R_cav,self.xi))
