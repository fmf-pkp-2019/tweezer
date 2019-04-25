import numpy as np
import scipy.constants

KB = scipy.constants.Boltzmann


def draw(k, temp, number_of_points):
    """Draws positions.

    Draws positions with uncorrelated coordinates from
    bivariate distribution, centered at (0, 0) and determined by
    tweezer coefficients and temperature.

    Parameters
    ----------
    k : tuple of floats
        tweezer coefficients (k_x, k_y)
    temp : float
        temperature in kelvins
    number_of_points : int
        number of positions to draw

    Returns
    -------
    xdata : ndarray
        x-coordinates
    ydata : ndarray
        y-coordinates

    Examples
    --------
    TODO

    """
    np.random.seed(seed=0)
    u1 = np.random.rand(number_of_points)
    u2 = np.random.rand(number_of_points)

    z1 = np.sqrt(-2. * np.log(u1)) * np.cos(2.*np.pi * u2)
    z2 = np.sqrt(-2. * np.log(u1)) * np.sin(2.*np.pi * u2)

    variance_x = KB*temp/k[0]*1e12
    variance_y = KB*temp/k[1]*1e12

    xdata = np.sqrt(variance_x) * z1
    ydata = np.sqrt(variance_y) * z2

    return xdata, ydata


def rotate_and_decenter(xdata, ydata, phi, center):
    """Rotates positions.

    Rotates positions by angle phi in anticlockwise direction
    and moves center of distribution.

    Parameters
    ----------
    xdata : ndarray
        x-coordinates
    ydata : ndarray
        y-coordinates
    phi : float
        angle of rotation
    center : tuple of floats
        (average x-coordinate, average y-coordinate)

    Returns
    -------
    new_xdata : ndarray
        new x-coordinates
    new_ydata : ndarray
        new y-coordinates

    Examples
    --------
    TODO

    """

    n = len(xdata)

    # Rotates one data point
    def rotate_one(phi, x, y):
        x_rotated = np.cos(phi)*x - np.sin(phi)*y
        y_rotated = np.sin(phi)*x + np.cos(phi)*y
        return x_rotated, y_rotated

    rotated_xdata, rotated_ydata = (np.zeros(n), np.zeros(n))
    for i in range(n):
        rotated_xdata[i], rotated_ydata[i] = rotate_one(
            phi, xdata[i], ydata[i]
        )
    return rotated_xdata + center[0], rotated_ydata + center[1]


def drift(xdata, ydata):
    """Adds a random sine function to the positions."""
    
    np.random.seed(seed=0)
    x_drift_parameters = np.random.rand(3)
    y_drift_parameters = np.random.rand(3)
    
    def sine(x, parameters):
        return parameters[0]*np.sin(parameters[1]*x/len(x)*5 + parameters[2])
    
    t = np.arange(len(xdata))

    return xdata + sine(t, x_drift_parameters), ydata + sine(t, y_drift_parameters)
    

def generate(k, temp=273, phi=0., center=(0., 0.), number_of_points=10**5):
    """Generates positions.

    Draws positions with uncorrelated coordinates from
    rotated bivariate distribution with moved center, determined by
    tweezer coefficients and temperature.

    Parameters
    ----------
    k : tuple of floats
        tweezer coefficients (k_x, k_y)
    temp : float
        temperature in kelvins
    phi : float
        angle of rotation
    center : tuple of floats
        (average x-coordinate, average y-coordinate)
    number_of_points : int
        number of positions to draw
    
    Returns
    -------
    xdata : ndarray
        x-coordinates
    ydata : ndarray
        y-coordinates

    Examples
    --------
    TODO

    """
    xdata, ydata = draw(k, temp, number_of_points)
    xdata, ydata = rotate_and_decenter(xdata, ydata, phi, center)
    xdata, ydata = drift(xdata, ydata)
    data = np.stack((xdata, ydata), axis=1)
    return data


def generate_time(number_of_points=10**5, time_interval=1e-3, start_time=0):
    """Generates time coordinates.
    
    Creates the time coordinates 
    with the specified number of points, 
    starting time and the interval between times.    

    Parameters
    ----------
    number_of_points : int
        number of positions to draw
    time_interval: float
        time between positions
    start_time: float
        time of the first position
    
    Returns
    -------
    tdata: ndarray
        time cordinates

    Examples
    --------
    TODO

    """
    return np.array(start_time + np.arange(number_of_points)*time_interval)
    
    
