#Script for extracting force values from dataset of optical tweezer measurements
import warnings

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