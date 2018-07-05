"""
Methods and functions to rotate or warp a pre-existing (3D) disk model


"""

import numpy as np


def rotate_disk(snapshot,theta,phi,dens_threshold = 0):
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
    pos = snapshot.gas.pos
    vel = snapshot.gas.vel
    
    # define rotation matrices
    mat1 = np.array([[1,0,0],\
                    [0,np.cos(theta),-np.sin(theta)],\
                    [0,np.sin(theta),np.cos(theta)]]).reshape(3,3)
    mat2 = np.array([[np.cos(phi),-np.sin(phi),0],\
                     [np.sin(phi),np.cos(phi),0],
                     [0,0,1]]).reshape(3,3)
    newpos = mat2.dot(mat1.dot(pos.T)).T
    newvel = mat2.dot(mat1.dot(vel.T)).T

    snapshot.gas.pos = newpos
    snapshot.gas.vel = newvel

    return snapshot
