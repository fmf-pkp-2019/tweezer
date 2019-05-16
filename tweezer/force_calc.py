#Script for extracting force values from dataset of optical tweezer measurements
import warnings
import math

import numpy as np
import scipy.constants as constants

def force_calculation(time, trajectory, trap_position, ks, temp=293):
    """Provided arrays of points in time and spatial coordinates of both the optical trap
    and the trapped particle as well as trap stiffnesses,
    
    the function calculates forces which the trap beam exerts on the particle (radially)
    and returns forces and their mean values

    Parameters
    ----------
    time : array_like
        time coordinates
    trajectory : ndarray_like
        x-coordinates and y-coordinates of trapped bead
    trap_position : ndarray_like
        x-coordinates and y-coordinates of trap
    ks : tuple of floats
        trap stiffnesses in x- and y-directions [N/m]
    temp : float
        system temperature [K]

    Note
    ----

    Returns
    -------
    forces : ndarray_like
        n-by-2 array of forces on bead in each time point
    means : array_like
        mean absolute values of forces (in x-,y-direction)

    Raises
    ------
    IndexError
        if sizes of any arrays differ from the others
    Warning
        if any trap coefficient is less than 0 (no bound state)
    """
    n = len(time)
    trajectory = np.array(trajectory)
    trap_position = np.array(trap_position)

    if ( n!=len(trajectory[:, 0]) or n!=len(trap_position[:, 0])):
        raise IndexError("Array dimensions need to be identical")
    if (temp < 0):
        raise ValueError("Verify temperature is converted to Kelvin")
    if (ks[0] < 0 or ks[1] < 0):
        warnings.warn("Value of one or more trap coefficients is negative")

    forces = np.zeros((n, 2))
    forces[:, 0] = ks[0]*(trajectory[:, 0] - trap_position[:, 0])
    forces[:, 1] = ks[1]*(trajectory[:, 1] - trap_position[:, 1])

    # Adjust to pN since position values are in micrometers
    means = np.mean(np.fabs(forces), axis=0)*1e6  
    print("Mean force values in pN:", means)

    return forces, means

def force_calculation_axis(time, trajectory, trap_position, ks_1, ks_2, temp=293):
    """Calculates forces acting on a pair of trapped particles along the trap-trap axis as a function of their distance,
    provided arrays of points in time and spatial coordinates of traps, particles, as well as trap stiffnesses.

    Note
    ----
    We assume that the trap axes are parallel to grid axes;
    the force components pointing along the trap-trap axis are then written into the 'forces' array and averaged.


    Parameters
    ----------
    time : array_like
        array of time values
    trajectory : ndarray_like
        x- and y-coordinates of trapped beads
    trap_position : ndarray_like
        x- and y-coordinates of traps
    ks_1 : tuple of floats
        stiffness of trap #1 in x- and y-directions [N/m]
    ks_2 : tuple of floats
        stiffness of trap #2 in x- and y-directions [N/m]
    temp : float
        system temperature [K]

    Returns
    -------
    forces : ndarray_like
        forces along the trap-trap axis on both particles for each point in time
    means : array_like
        means of those forces
    distance : float
        distance between traps

    Examples
    --------
    TODO
    """

    n = len(time)
    
    if ( n!=len(trajectory[:, 0]) or n!=len(trap_position[:, 0])):
        raise IndexError("Array dimensions need to be identical")

    forces = np.zeros((n,2))

    for point in range(n):
        if (trap_position[point,2] == trap_position[point,0]):
            if (trap_position[point,3] - trap_position[point,1] > 0):
                phi = math.pi/2.
            elif (trap_position[point,3] - trap_position[point,1] < 0):
                phi = -math.pi/2.
        else:
            phi = np.arctan((trap_position[point,3] - trap_position[point,1])/(trap_position[point,2] - trap_position[point,0]))
            
        theta_1 = np.arctan((trajectory[point,1]-trap_position[point,1])/(trajectory[point,0]-trap_position[point,0]))
        theta_2 = np.arctan((trajectory[point,3]-trap_position[point,3])/(trajectory[point,2]-trap_position[point,2]))
        keff_1 = math.sqrt((ks_1[0]*math.cos(theta_1))**2 + (ks_1[1]*math.sin(theta_1))**2)
        keff_2 = math.sqrt((ks_2[0]*math.cos(theta_2))**2 + (ks_2[1]*math.sin(theta_2))**2)
        r_1 = math.sqrt((trajectory[point,0]-trap_position[point,0])**2 + (trajectory[point,1]-trap_position[point,1])**2)
        r_2 = math.sqrt((trajectory[point,2]-trap_position[point,2])**2 + (trajectory[point,3]-trap_position[point,3])**2)

        forces[point,0] = keff_1*r_1*math.cos(theta_1 - phi)
        forces[point,1] = keff_2*r_2*math.cos(theta_2 - phi - math.pi)

    means = np.mean(np.fabs(forces), axis=0)*1e6
    # adjusted to pN since position values are in micrometers
    distance = math.sqrt((trap_position[0,0]-trap_position[0,2])**2 + (trap_position[0,1]-trap_position[0,3])**2)

    return forces, means, distance

def sigma_calculation(trajectory, trap_position):
    """Provided arrays of optical trap positions and particle trajectories, calculates the mean displacements and their uncertainties.

    Parameters
    ----------
    trajectory : ndarray_like
        x- and y- positions of both particles
    trap_position : ndarray_like
        x- and y- positions of both optical traps

    Returns
    -------
    means : array_like
        mean displacements of particles from trap centre in x- and y-directions
    sigmas : array_like
        uncertainties of displacements in x- and y-directions
    """

    n,m = trajectory.shape
    if (n!=len(trap_position[:, 0])):
        raise IndexError("Number of array elements needs to be identical")
    elif (trap_position.shape[1] < m):
        raise IndexError("Trap coordinates undefined for %d columns" % (m-trap_position.shape[1]))

    displacements = np.zeros((n,m))
    squares = np.zeros((n,m))
    sigmas = np.zeros(m)

    for point in range(n):
        for i in range(m):
            displacements[point,i] = trajectory[point,i]-trap_position[point,i]
            squares[point,i] = displacements[point,i]**2
    
    means = np.mean(np.fabs(displacements), axis=0)
    means_sq = np.mean(np.fabs(squares), axis=0)

    for point in range(m):
        sigmas[point] = math.sqrt(abs(means_sq[point] - np.square(means[point])))

    return means, sigmas
