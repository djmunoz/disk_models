import numpy as np
import matplotlib.pyplot as plt

# 3D Kernel coefficients
KERNEL_COEFF_1 = 2.546479089470   # (8/PI)
KERNEL_COEFF_2 = 15.278874536822  # (48/PI)
KERNEL_COEFF_3 = 45.836623610466  # (3*48/PI) 
KERNEL_COEFF_4 = 30.557749073644  # (2*48/PI)
KERNEL_COEFF_5 = 5.092958178941   # (16/PI)
KERNEL_COEFF_6 = -15.278874536822 # (-48/PI)
NORM_COEFF =  4.188790204786 # (4/3*PI)


def W(R,h):
   
    h_inv = 1.0 / h
    h3_inv = h_inv**3
    u = R * h_inv
    wk = u * 0
    
    if (isinstance(R,(list, tuple, np.ndarray))):
        ind1 = u < 0.5
        ind2 = (u >= 0.5) & (u < 1.0)
        ind3 = u >= 1.0
        wk[ind1] = h3_inv * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * u[ind1]**2 *(u[ind1] - 1))
        wk[ind2] = h3_inv * KERNEL_COEFF_5 * (1 - u[ind2] )**3
        wk[ind3] = 0
    else:
        if(R < h):
            if(u < 0.5):
                wk = h3_inv * (KERNEL_COEFF_1 + KERNEL_COEFF_2 * u**2 *(u - 1))
            elif (u < 1.0):
                wk = h3_inv * KERNEL_COEFF_5 * (1 - u)**3
        else:
            wk = 0
                
    return wk


def Wprime(R,h):
   
    h_inv = 1.0 / h
    h4_inv = h_inv**4
    u = R * h_inv
    dwk = u * 0
    
    if (isinstance(R,(list, tuple, np.ndarray))):
        ind1 = u < 0.5
        ind2 = (u >= 0.5) & (u < 1.0)
        ind3 = u >= 1.0
        dwk[ind1] = h4_inv * u[ind1] * (KERNEL_COEFF_3 * u[ind1] - KERNEL_COEFF_4) 
        dwk[ind2] = h4_inv * KERNEL_COEFF_6 * (1 - u[ind2])**2
        dwk[ind3] = 0
    else:
        if(R < h):
            if(u < 0.5):
                dwk = h4_inv * u * (KERNEL_COEFF_3 * u - KERNEL_COEFF_4)  
            elif (u < 1.0):
                dwk = h4_inv * KERNEL_COEFF_6 * (1 - u)**2
        else:
            dwk = 0
                
    return dwk 

def g(R,h):

    h_inv = 1.0 / h
    u = R * h_inv
    fac = u * 0
    
    if (isinstance(R,(list, tuple, np.ndarray))):
        ind1 = u < 0.5
        ind2 = (u >= 0.5) & (u < 1.0)
        ind3 = u >= 1.0
        fac[ind1] = h_inv * (-14.0 / 5 + u[ind1] **2 * (16.0/ 3 + u[ind1] **2*(-48.0 /5  + 32.0/5 * u[ind1] )))
        fac[ind2] = h_inv * (1.0/15.0 / u[ind2]  - 16.0 /5 + \
                             u[ind2] **2 *(32.0 /3 + u[ind2]  * \
                                           (-16 + 48.0/5 * u[ind2]  - 32.0/15 * u[ind2]**2)))
        fac[ind3] = -h_inv / u[ind3]
    else:
        if(R < h):
            if(u < 0.5):
                fac = h_inv * (-14.0 / 5 + u**2 * (16.0/ 3 + u**2*(-48.0 /5  + 32.0/5 * u)))
            elif (u < 1.0):
                fac = h_inv * (1.0/15.0 / u - 16.0 /5 + \
                               u**2 *(32.0 /3 + u * (-16 + 48.0/5 * u - 32.0/15 * u**2)))
        else:
            fac = -h_inv / u
                
                
    return -fac

def g1(R,h):

    h_inv = 1.0 / h
    h3_inv = h_inv**3
    u = R * h_inv
    fac = u * 0
    
    if (isinstance(R,(list, tuple, np.ndarray))):
        ind1 = u < 0.5
        ind2 = (u >= 0.5) & (u < 1.0)
        ind3 = u >= 1.0
        fac[ind1] = h3_inv * (-32.0 / 3 + u[ind1]**2 * (192.0/ 5 - 32 * u[ind1]))
        fac[ind2] = h3_inv * (1.0/15.0 / u[ind2]**3 - 64.0 / 3 + 48 * u[ind2] \
                               -192.0 /5 * u[ind2]**2 + 32.0 /3 * u[ind2]**3)
        fac[ind3] = -h3_inv / u[ind3]**3
    else:
        if(R < h):
            if(u < 0.5):
                fac = h3_inv * (-32.0 / 3 + u**2 * (192.0/ 5 - 32 * u))
            elif (u < 1.0):
                fac = h3_inv * (1.0/15.0 / u**3 - 64.0 / 3 + 48 * u
                                -192.0 /5 * u**2 + 32.0 /3 * u**3)
        else:
            fac = -h3_inv / u**3
                
                
    return -fac



def g2(R,h):

    h_inv = 1.0 / h
    h5_inv = h_inv**5
    u = R * h_inv
    fac = u * 0
    
    if (isinstance(R,(list, tuple, np.ndarray))):
        ind1 = u < 0.5
        ind2 = (u >= 0.5) & (u < 1.0)
        ind3 = u >= 1.0
        fac[ind1] = h5_inv * (384.0/5 - 96 * u[ind1])
        fac[ind2] = h5_inv * (-384.0/5 - 0.2 /u[ind2] **5 + 48.0 / u[ind2]  + 32.0  *u[ind2] )
        fac[ind3] = h5_inv * 3 / u[ind3]**5
    else:
        if(R < h):
            if(u < 0.5):
                fac = h5_inv * (384.0/5 - 96 * u)
            elif (u < 1.0):
                fac = h5_inv * (-384.0/5 - 0.2 /u**5 + 48.0 / u + 32.0 * u)
        else:
            fac = h5_inv * 3 / u**5
                
                
    return fac / 3

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

    
if __name__ == '__main__':

    r = np.linspace(0.001,2.0,100)
    plt.plot(r,1.0/r,'k--',label=r'$1/r$')
    plt.plot(r,1.0/r/r,'k:',label=r'$1/r^2$')
    plt.plot(r,1.5/r**2.5,'-.',color='purple',label=r'$(3/2)/r^{5/2}$')
    plt.plot(r,1.0/r**1.5,'-.',color='orange',label=r'$1/r^{3/2}$')
    plt.plot(r,1.0/r**0.5,'-.',color='lightblue',label=r'$1/r^{1/2}$')
    
    plt.plot(r,g(r,1.0),label=r'$g$')
    #plt.plot(r,-np.gradient(g(r,1.0))/np.gradient(r),'k:')
    plt.plot(r,r**1.5*g1(r,1.0),label=r'$rg_1$',lw=3.0,alpha=0.4)
    plt.plot(r,-np.gradient(np.sqrt(g1(r,1.0)))/np.gradient(r),
             color='purple',label=r"$-g_1'$",lw=3.0,alpha=0.4)
    plt.plot(r,r**2*np.sqrt(g2(r,1.0)),label=r'$r^2g_2$')
    plt.plot(r,r*g2(r,1.0)/np.sqrt(g1(r,1.0))/2,label=r'$d\Omega/dr$')
    #plt.plot(r,g3(r,1.0),label=r'$g_3$')
    #plt.plot(r,g4(r,1.0),label=r'$g_4$')
    plt.legend()
    plt.ylim(0,7)
    plt.show()


    plt.plot(r,W(r,1.0))
    plt.plot(r,-Wprime(r,1.0))
    plt.show()
