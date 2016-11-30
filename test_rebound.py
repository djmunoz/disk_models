import rebound
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    sim = rebound.Simulation()
    sim.add(m=0.5,x=24.35963491,y=25.30757223,z= 25.0,
            vx=-1.90261535e-01,vy=-5.06786878e-01,vz=-3.31455694e-11)
    sim.add(m=0.5,x=25.30641956,y=24.98570452,z=25.0,
            vx=1.31606185e-01,vy=4.39997773e-01,vz=2.43159375e-11)
    sim.add(m=0.2,x=27.08715951,y=23.16702034,z=25.0,
            vx=3.66595937e-01,vy=4.17431910e-01,vz=5.51851996e-11) 
    sim.move_to_com()
    
    #sim.integrator = "whfast"
    sim.integrator = "ias15"
    sim.dt = 1e-3

    torb = 2.*np.pi * (3)**1.5 
    Noutputs = 30000
    times = np.linspace(0, 30.*torb, Noutputs)
    x0 = np.zeros(Noutputs)
    y0 = np.zeros(Noutputs)
    x1 = np.zeros(Noutputs)
    y1 = np.zeros(Noutputs)
    x2 = np.zeros(Noutputs)
    y2 = np.zeros(Noutputs)
    for i,time in enumerate(times):
        sim.integrate(time, exact_finish_time=0)
        x0[i] = sim.particles[0].x
        y0[i] = sim.particles[0].y
        x1[i] = sim.particles[1].x
        y1[i] = sim.particles[1].y
        x2[i] = sim.particles[2].x
        y2[i] = sim.particles[2].y

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    plt.plot(x0, y0)
    plt.plot(x1, y1)
    plt.plot(x2, y2,color='red')
    plt.show()
