import numpy as np


def g3(R,h):

    h_inv = 1.0 / h
    h7_inv = h_inv**7
    u = R * h_inv
    fac = u * 0
    
    if (isinstance(R,(list, tuple, np.ndarray))):
        ind1 = u < 0.5
        ind2 = (u >= 0.5) & (u < 1.0)
        ind3 = u >= 1.0
        fac[ind1] = h7_inv * (- 96.0) / u[ind1]
        fac[ind2] = h7_inv * (32.0 / u[ind2] + 1.0 / u[ind2]**7
                              - 48.0 / u[ind2]**3)
                    
        fac[ind3] = h7_inv * (-15.0) / u[ind3]**7
    else:
        if(R < h):
            if(u < 0.5):
                fac = h7_inv * (- 96.0) / u
            elif (u < 1.0):
                fac = h7_inv * (32.0 / u + 1.0 / u**7 - 48.0 / u**3)
        else:
            fac = h7_inv * (-15.0) / u**7
                
                
    return fac/(-15)

def g4(R,h):

    h_inv = 1.0 / h
    h5_inv = h_inv**5
    u = R * h_inv
    fac = u * 0
    if (isinstance(R,(list, tuple, np.ndarray))):
        ind1 = u < 0.5
        ind2 = (u >= 0.5) & (u < 1.0)
        ind3 = u >= 1.0
        fac[ind1] = h5_inv * 19.2 * (4 - 5 * u[ind1])
        fac[ind2] = h5_inv * (48.0 / u[ind2] - 0.2 / u[ind2]**5 -
                              76.8 + 32 * u[ind2])
        fac[ind3] = h5_inv * 3  / u[ind3]**5
    else:
        if(R < h):
            if(u < 0.5):
                fac = h5_inv * 19.2 * (4 - 5 * u)
            elif (u < 1.0):
                fac = h5_inv * (48.0 / u - 0.2 / u**5 - 76.8 + 32 * u)
        else:
            fac = h5_inv * 3  / u**5
                
                
    return fac/3

    
