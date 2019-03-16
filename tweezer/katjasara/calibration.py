import numpy as np
import scipy.constants

import matplotlib.pyplot as plt


KB = scipy.constants.Boltzmann


def subtract_moving_average(time, data, averaging_time):
    """Subtracts moving average from data.

    Assumes data points are spaced evenly in time. Computes moving average
    by averaging data over time interval of width averaging_time
    and subtracts it from data.

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
        data with moving average subtracted

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

    moving_average = np.zeros(len(data))
    for i in range(len(moving_average)):
        if i < n:
            moving_average[i] = np.mean(data[0:i+n])
        elif i < len(data) - n:
            moving_average[i] = np.mean(data[i-n:i+n])
        else:
            moving_average[i] = np.mean(data[i-n:len(data)])

    return data - moving_average


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
    new_xdata : ndarray
        new x-coordinates
    new_ydata : ndarray
        new y-coordinates
    phi : float
        Angle in anticlockwise direction by which positions were rotated.
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
    if len(xdata) != len(ydata):
        raise ValueError("Unclear number of points.")

    def center(data):
        return data - np.mean(data)

    def rotate(xdata, ydata):
        n = len(xdata)

        cov = np.cov(xdata, ydata)
        var, vec = np.linalg.eigh(cov)
        phi = -np.arctan(vec[:, 1][1]/vec[:, 1][0])

        # Rotates one data point
        def rotate_one(phi, x, y):
            x_rotated = np.cos(phi)*x - np.sin(phi)*y
            y_rotated = np.sin(phi)*x + np.cos(phi)*y
            return x_rotated, y_rotated

        new_xdata, new_ydata = (np.zeros(n), np.zeros(n))
        for i in range(n):
            new_xdata[i], new_ydata[i] = rotate_one(
                phi, xdata[i], ydata[i]
            )
        return new_xdata, new_ydata, phi, var[::-1]

    return rotate(center(xdata), center(ydata))


def calibrate(time, xdata, ydata, averaging_time=10., temp=293.):
    """Calibrates laser tweezer.

    Subtracts moving average from xdata and ydata,
    centers xdata and ydata and rotates positions so that k_x < k_y.
    Computes k_x and k_y from variances.

    Parameters
    ----------
    time : array_like
        time coordinates
    xdata : array_like
        x-coordinates
    ydata : array_like
        y-coordinates
    averaging_time : float
        averaging time interval
    temp : float
        temperature in kelvins

    Returns
    -------
    k : tuple of floats
        laser tweezer coefficients (k_x, k_y)
    phi : float
        Angle in anticlockwise direction by which positions were rotated.

    Examples
    --------
    TODO

    """
    x = subtract_moving_average(time, xdata, averaging_time)
    y = subtract_moving_average(time, ydata, averaging_time)

    x, y, phi, var = center_and_rotate(x, y)
    k = KB*temp/var*1e12

    return tuple(k), phi


def plot(time, xdata, ydata, averaging_time=10., temp=293.):
    """
    TODO (will be part of another module later)

    """
    x = subtract_moving_average(time, xdata, averaging_time)
    y = subtract_moving_average(time, ydata, averaging_time)
    x, y, phi, var = center_and_rotate(x, y)
    k = KB*temp/var*1e12

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original data')
    ax1.set_xlabel('x [10^(-6) m]')
    ax1.set_ylabel('y [10^(-6) m]')
    ax1.scatter(xdata, ydata, s=4)
    ax1.set_aspect('equal')
    ax2.set_title('Centered data, phi = {:.2f} rad'.format(phi,))
    ax2.set_xlabel('x [10^(-6) m]')
    ax2.set_ylabel('y [10^(-6) m]')
    ax2.scatter(x, y, s=4)
    ax2.set_aspect('equal')
    fig.tight_layout()
    plt.show()

    xy = np.zeros((len(x), 2))
    xy[:, 0] = x
    xy[:, 1] = y

    fig = plt.figure()
    titles = ['x', 'y']
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        ax.set_xlabel('x [10^(-6) m]')
        ax.set_ylabel('Bin height')
        hist, bin_edges = np.histogram(
            xy[:, i], bins=int(np.sqrt(len(x))), density=True
        )
        ax.set_title('k_{} = {:.2e}J/m^2'.format(
            titles[i], k[i]))
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
        ax.scatter(bin_centres, hist, s=4)
        x_model = np.linspace(min(bin_centres), max(bin_centres), 100)
        prefactor = 1./np.sqrt(2.*np.pi*var[i])
        ax.plot(x_model, prefactor*np.exp(-x_model**2./(2.*var[i])))
    fig.tight_layout()
    plt.show()

    return None
