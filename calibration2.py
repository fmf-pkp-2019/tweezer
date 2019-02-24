import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt


def subtract_linear_drift(t, x):
    """
    Subtracts the linear drift from positions.

    Args:
        t: Array of times.
        x: Array of positions.
        
    Returns:
        Array of positions without the linear drift.
    """
    
    def linear(x, k, c):
        return k*x+c
    
    [k, c], _ = scipy.optimize.curve_fit(linear, t, x)
    return x - linear(t, k, c)


def center_and_rotate(xdata, ydata):
    """
    Sets the average position as the origin,
    makes the major axis of the ellipse lie on the x-axis.

    Args:
        xdata: Array of x-positions.
        ydata: Array of y-positions.
    
    Returns:
        Angle of the counterclochwise rotation (in radians),
        array of rotated x-positions,
        array of rotated y-positions.
    """

    if len(xdata) != len(ydata):
        raise ValueError("Unclear number of points.")
    
    def center(data):
        return data - np.mean(data)

    xdata_centered = center(xdata)
    ydata_centered = center(ydata)

    def linear(x, k):
        return k*x

    k, _ = scipy.optimize.curve_fit(
        linear, xdata_centered, ydata_centered
        )
    phi = -np.arctan(k)

    def rotate(phi, x, y):
        x_rotated = np.cos(phi)*x - np.sin(phi)*y
        y_rotated = np.sin(phi)*x + np.cos(phi)*y
        return x_rotated, y_rotated

    xy_rotated = np.zeros((len(xdata), 2))
    for i in range(len(xy_rotated)):
        xy_rotated[i] = rotate(
            phi, xdata_centered[i], ydata_centered[i]
            )

    return phi, xy_rotated[:, 0], xy_rotated[:, 1]


def histogram_and_fit_gaussian(data):
    """
    Calibrates by histogramming position deviations (into 30 bins),
    and fitting a gaussian function y = a e^(-x^2/(2*variance).

    Args:
        data: Array of position deviations in one dimension.

    Returns:
        Centres of histogram bins,
        histogram values,
        a and variance.
    """
    heights, bin_edges = np.histogram(
        data, bins = 30,
    )
    sum_heights = len(data)
    heights = heights/sum_heights
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

    def gauss(x, prefactor, variance):
        return prefactor*np.exp(-x**2/(2.*variance))

    (prefactor, variance), _ = scipy.optimize.curve_fit(
        gauss, bin_centres, heights,
        p0 = [0.1, np.var(data)]
    )

    return bin_centres, heights, prefactor, variance

    
def histogram_and_fit_quadratic(data):
    """
    Calibrates by histogramming position deviations
    (into 30 bins),
    computing the natural logarithm of the heights and
    fitting a quadratic function y = a x^2 - b.

    Args:
        data: Array of position deviations in one dimension.

    Returns:
        Centres of histogram bins,
        -log(bin heights),
        a and b.
    """

    heights, bin_edges = np.histogram(
        data, bins = 30,
        density = True
    )
    sum_heights = float(len(data))
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    
    ok_index = heights > 0
    heights = heights[ok_index]
    bin_centres = bin_centres[ok_index]

    def quadratic(x, prefactor, constant):
        return prefactor*x**2. - constant

    y = -np.log(heights/sum_heights)
    (prefactor, constant), _ = scipy.optimize.curve_fit(
        quadratic, bin_centres, y
    )

    return bin_centres, y, prefactor, constant


def calibrate1(xdata, ydata, temperature = 293):
    """
    Calibrates a tweezer using histogram_and_fit_quadratic.

    Args:
        xdata: Array of x-positions.
        ydata: Array of y-positions.
    """
    
    phi, x, y = center_and_rotate(xdata, ydata)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original data')
    ax1.set_xlabel('x [10^(-6) m]')
    ax1.set_ylabel('y [10^(-6) m]')
    ax1.hist2d(xdata, ydata, bins = 30)
    ax1.set_aspect('equal')
    ax2.set_title('Centered rotated data \n phi = {:.3f}'.format(*phi,))
    ax2.set_xlabel('x [10^(-6) m]')
    ax2.set_ylabel('y [10^(-6) m]')
    ax2.hist2d(x, y, bins = 30)
    ax2.set_aspect('equal')
    fig.tight_layout()
    plt.show()

    xy = np.zeros((len(xdata), 2))
    xy[:, 0] = x
    xy[:, 1] = y
    
    """Density with Gaussian fit."""
    fig = plt.figure()
    titles = ['x', 'y']
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        ax.set_xlabel('{} [10^(-6) m]'.format(titles[i]))
        ax.set_ylabel('Denstity')
        bin_centres, y, a, var = histogram_and_fit_gaussian(
            xy[:, i]
            )
        ax.set_title('k_{} = {:.2e} J/m^2'.format(
            titles[i], 1e12*1.38064852e-23*temperature/var))
        ax.scatter(bin_centres, y)
        x_model = np.linspace(min(bin_centres), max(bin_centres), 100)
        ax.plot(x_model, a*np.exp(-x_model**2/(2*var)))
    fig.tight_layout()
    plt.show()
    
    """Potential with quadratic fit."""
    fig = plt.figure()
    titles = ['x', 'y']
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        ax.set_xlabel('x [10^(-6) m]')
        ax.set_ylabel('Potential [kT]')
        bin_centres, y, a, b = histogram_and_fit_quadratic(
            xy[:, i]
            )
        ax.set_title('k_{} = {:.2e} J/m^2'.format(
            titles[i], 2*a*1e12*1.38064852e-23*temperature))
        ax.scatter(bin_centres, y)
        x_model = np.linspace(min(bin_centres), max(bin_centres), 100)
        ax.plot(x_model, a*x_model**2. - b)
    fig.tight_layout()
    plt.show()

    return None