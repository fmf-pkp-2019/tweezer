#Script for extracting force values from dataset of optical tweezer measurements
import warnings

import matplotlib.pyplot as pyplot
import numpy as np
import scipy.constants as constants

def read_input(filename):
    """Placeholder function - must be adjusted for reading actual experimentally generated data

    Parameters
    ----------
    filename : str
        name of file to open

    Returns
    -------
    temporary : np.array of float
        five-column array of data read from file
    """

    temporary = np.loadtxt(
        filename,
        delimiter ="\t",
        usecols=np.r_[range(0,1), range(2,4), range(14,16)],
        dtype=np.float64
        )
    return temporary


def force_plotting(time, forces):
    """Creates a basic F(t) plot.

    Parameters
    ----------
    time : list of float
        times of recorded points
    forces : np.array of float
        two-column array of forces on trapped bead in x- and y-directions
    """        

    pyplot.plot(time, forces[:,0]*1e6, "r", linewidth = 0.5, label="$F_x$")
    pyplot.plot(time, forces[:,1]*1e6, "b", linewidth = 0.5, label="$F_y$")
    pyplot.plot(time, np.sqrt(forces[:,0]**2 + forces[:,1]**2)*1e6, "k", linewidth = 0.5, label="$F_{sum}$")
    pyplot.title("Radial gradient forces on trapped particle")
    pyplot.xlabel("t [s]")
    pyplot.ylabel("F [pN]")
    pyplot.grid(True)
    pyplot.legend(loc='best')
    pyplot.show()

    return

def force_calculation(time, pos_x, pos_y, trap_pos_x, trap_pos_y, ks, temp=293):
    """Provided arrays of points in time and spatial coordinates of both optical trap
    and trapped particle as well as trap stiffnesses,
    
    the function calculates forces which the trap beam exerts on the particle (radially)
    and returns forces and their mean values

    Parameters
    ----------
        time : list of float
            times of 
        pos_x : list of float
            x-positions of trapped bead
        pos_y : list of float
            y-positions of trapped bead
        trap_pos_x : list of float
            x-positions of trap
        trap_pos_y : list of float
            y-positions of trap
        ks : tuple of float
            trap stiffnesses in x- and y-directions [N/m]
        temp : float
            system temperature [K]

    Note
    ----

    Returns
    -------
    forces : list of float
        n-by-2 array of forces on bead in each time point
    means : tuple of float
        mean absolute values of forces (in x-,y-direction)

    Raises
    ------
    IndexError
        if sizes of any arrays differ from the others
    Warning
        if any trap coefficient is less than 0 (no bound state)
    """

    n = len(time)

    if ( n!=len(pos_x) or n!=len(pos_y) or n!=len(trap_pos_x) or n!=len(trap_pos_y)):
        raise IndexError("Array dimensions need to be identical")
    if (temp < 0):
        raise ValueError("Verify temperature is converted to Kelvin")
    if (ks[0] < 0 or ks[1] < 0):
        warnings.warn("Value of one or more trap coefficients is negative")

    forces = np.zeros((n,2))

    for point in range(n):
        forces[point,0] = ks[0]*(pos_x[point]-trap_pos_x[point])
        forces[point,1] = ks[1]*(pos_y[point]-trap_pos_y[point])

    means = np.mean(np.fabs(forces), axis=0)*1e6  # Adjust to pN since position values are in micrometers
    print("Mean force values in pN:", means)

    return forces, means
