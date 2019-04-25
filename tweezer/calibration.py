import numpy as np
import scipy.constants

import matplotlib.pyplot as plt

KB = scipy.constants.Boltzmann

def subtract_moving_average(time, data, averaging_time):
    """Subtracts moving average from data.

    Assumes data points are spaced evenly in time. Computes moving average
    by averaging data over time interval of width averaging_time
    and subtracts it from data.
    The data is shortened to the valid range, so there are no data 
    boundary effects.

    Parameters
    ----------
    time : array_like
        time coordinates
    data : array_like
        data to subtract moving average from
    averaging_time :
        averaging time interval

    Returns
    -------
    new_data : ndarray
        shortened data with moving average subtracted
    moving_average: ndarray
        moving average
    new_time: ndarray
        dhorrtened time array

    Raises
    ------
    ValueError
        if dimensions of time and data do not match or
        if averaging_time is too short.

    Examples
    --------
    TODO

    """
    if len(time) != len(data):
        raise ValueError("Unclear number of points.")

    # Average time interval between successive data points
    dt = (time[-1] - time[0])/len(time)
    # Half the number of data points to compute moving average from
    n = int(averaging_time/dt/2.)

    if n == 0:
        raise ValueError("Too short averaging time.")
    elif n > len(data)/2.:
        raise ValueError("Too long averaging time.")

    moving_average = np.convolve(data, np.ones(((int)(2*n),))/(float)(2*n), mode = 'valid')
    new_data = data[n:-n+1] - moving_average
    new_time = time[n:-n+1]

    return new_data, moving_average, new_time


def center_and_rotate(xdata, ydata):
    """Centers and rotates positions.

    Centers positions by subtracting average position.
    Assumes distribution of positions is bivariate,
    makes major axis of ellipse lie on x-axis
    by diagonalization of correlation matrix.

    Parameters
    ----------
    xdata : array_like
        x-coordinates
    ydata : array_like
        y-coordinates   

    Returns
    -------
    rotated_data : ndarray
        new x-coordinates and y-coordinates
    phi : float
        angle in anticlockwise direction by which positions were rotated
    var : ndarray
        new variances

    Raises
    ------
    ValueError
        if dimensions of xdata and ydata do not match.

    Examples
    --------
    TODO

    """
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    
    if len(xdata) != len(ydata):
        raise ValueError("Unclear number of points.")

    def center(data):
        return data - np.mean(data)

    def rotate(xdata, ydata):
        cov = np.cov(xdata, ydata)
        var, vec = np.linalg.eigh(cov)
        phi = -np.arctan(vec[:, 1][1]/vec[:, 1][0])
        rotated_data = np.empty((len(xdata), 2))
        rotated_data[:, 0] = np.cos(phi)*xdata - np.sin(phi)*ydata
        rotated_data[:, 1] = np.cos(phi)*ydata + np.sin(phi)*xdata
        return rotated_data, phi, var[::-1]

    return rotate(center(xdata), center(ydata))


def calibrate(time, data, averaging_time=1., temp=293.):
    """Calibrates tweezer.

    Subtracts moving average from xdata and ydata,
    centers xdata and ydata and rotates positions so that k_x < k_y.
    Computes k_x and k_y from variances.

    Parameters
    ----------
    time : array_like
        time coordinates
    data : ndarray_like
        x-coordinates and y-coordinates
    averaging_time : float
        averaging time interval
    temp : float
        temperature in kelvins

    Returns
    -------
    ks : tuple of floats
        trap stiffnesses in x- and y-directions [N/m]
    phi : float
        angle in anticlockwise direction by which positions were rotated
    data : ndarray_like
        x-coordinates and y-coordinates of averaged data

    Examples
    --------
    TODO

    """
    data = np.array(data)
    x, x_average = subtract_moving_average(time, data[:, 0], averaging_time)[:2]
    y, y_average = subtract_moving_average(time, data[:, 1], averaging_time)[:2]
    trajectory, phi, var = center_and_rotate(x, y)
    ks = KB*temp/var*1e12

    return tuple(ks), phi, np.array([x_average, y_average])
  
def potential(time, data, averaging_time=1., temp=293.):
    """Calculates the potential.

    Centers and rotates data. Histogramms the data
    to get the probability denisity.
    Computes the potential in units of kBT as log(rho).

    Parameters
    ----------
    time : array_like
        time coordinates
    data : ndarray_like
        x-coordinates and y-coordinates
    averaging_time : float
        averaging time interval
    temp : float
        temperature in kelvins

    Returns
    -------
    positions: list of two arrays
        x- and y-coordinates
    potential_values: list of two arrays
        values of the potential coresponding to the postitions
    phi: float
        angle in anticlockwise direction by which positions were rotated

    Examples
    --------
    TODO

    """
    data = np.array(data)
    x = subtract_moving_average(time, data[:, 0], averaging_time)[0]
    y = subtract_moving_average(time, data[:, 1], averaging_time)[0]
    trajectory, phi, var = center_and_rotate(x, y)
    
    positions = [0, 0]
    potential_values = [0, 0]
    
    for i in range(2):
       hist, bin_edges = np.histogram(trajectory[:, i], bins=int(np.sqrt(len(x))), density=False)
       hist = hist/(float)(len(x))
       bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
       ok_index = hist > 0
       hist, bin_centres = hist[ok_index], bin_centres[ok_index]
       positions[i] = bin_centres
       potential_values[i] = -np.log(hist)

    return positions, potential_values, phi