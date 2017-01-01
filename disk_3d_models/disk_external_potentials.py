import numpy as np

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


   
def spherical_potential_keplerian(r,soft=0.01):
    return SplineProfile(r,soft)


