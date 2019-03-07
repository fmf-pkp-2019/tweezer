#Script for extracting force values from dataset of optical tweezer measurements

import matplotlib.pyplot as pyplot
import numpy as np
import scipy.constants as constants
import warnings


def read_input(filename):

    #   Placeholder, must be adjusted for reading actual experimentally generated data

    temporary = np.loadtxt(filename, usecols=range(1,6), dtype=np.float64)
    return temporary


def force_plotting(time, forces):

    #   creates a basic F(t) plot

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

def force_calculation(time, xPos, yPos, xPosTrap, yPosTrap, ks, temp=293):

    #   Provided arrays of time points, spatial positions of both the optical trap and trapped particle, trap stiffnesses:
    #   -calculates and plots forces the trap beam exerts on the particle (radially)
    #   -returns a n-by-2 array of forces and a tuple of mean (absolute) force values

    n = len(time)

    if ( n!=len(xPos) or n!=len(yPos) or n!=len(xPosTrap) or n!=len(yPosTrap)):
        raise IndexError("Array dimensions need to be identical")
    if (temp < 0):
        raise ValueError("Verify temperature is converted to Kelvin")
    if (ks[0] < 0 or ks[1] < 0):
        warnings.warn("Value of one or more trap coefficients is negative")

    forces = np.zeros((n,2))

    for point in range(n):
        forces[point,0] = ks[0]*(xPos[point]-xPosTrap[point])
        forces[point,1] = ks[1]*(yPos[point]-yPosTrap[point])

    means = np.mean(np.fabs(forces), axis=0)*1e6
    print("Mean force values in pN:", means)

    return forces, means