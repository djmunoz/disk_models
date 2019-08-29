from __future__ import print_function
"""
Methods and functions to rotate or warp a pre-existing (3D) disk model


"""

import numpy as np


def rotate_disk(pos,vel,theta,phi,dens_threshold = 0):
    """
    Rotate, rigidly, a disk model by an azimuthal angle phi (defined
    respect to the original x-axis) and polar angle theta (defined
    respect to the original z-axis)

    
    Parameters
    ----------
    first : 
    second :
    third : 
    
    Returns
    -------
    
    Raises
    ------

    """

    # extract positions and velocities
    pos = snapshot.gas.pos.copy()
    vel = snapshot.gas.vel.copy()
    
    # define rotation matrices
    mat1 = np.array([[1,0,0],\
                    [0,np.cos(theta),-np.sin(theta)],\
                    [0,np.sin(theta),np.cos(theta)]]).reshape(3,3)
    mat2 = np.array([[np.cos(phi),-np.sin(phi),0],\
                     [np.sin(phi),np.cos(phi),0],
                     [0,0,1]]).reshape(3,3)
    R0 = (pos[:,:]).sum(axis=0)
    pos[:,0]-=R0[0]
    pos[:,1]-=R0[1]
    pos[:,2]-=R0[2]
    # Rotate positions and velocities
    newpos = mat2.dot(mat1.dot(pos.T)).T
    newvel = mat2.dot(mat1.dot(vel.T)).T
    newpos[:,0]+=R0[0]
    newpos[:,1]+=R0[1]
    newpos[:,2]+=R0[2]
    
    snapshot.gas.pos = newpos
    snapshot.gas.vel = newvel

    return snapshot
