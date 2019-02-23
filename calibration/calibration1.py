import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt

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


def histogram_and_fit_quadratic(data):
    """
    Calibrates by histogramming position deviations
    (into 20 bins),
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
        data, bins = 20,
        density = True
    )
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

    def quadratic(x, prefactor, constant):
        return prefactor*x**2. - constant

    y = -np.log(heights)
    (prefactor, constant), _ = scipy.optimize.curve_fit(
        quadratic, bin_centres, y
    )

    return bin_centres, y, prefactor, constant


def calibrate1(xdata, ydata):
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
    ax1.scatter(xdata, ydata, s = 4)
    ax1.set_aspect('equal')
    ax2.set_title('Centered data, phi = {:.3f}'.format(*phi,))
    ax2.set_xlabel('x [10^(-6) m]')
    ax2.set_ylabel('y [10^(-6) m]')
    ax2.scatter(x, y, s = 4)
    ax2.set_aspect('equal')
    fig.tight_layout()
    plt.show()

    xy = np.zeros((len(xdata), 2))
    xy[:, 0] = x
    xy[:, 1] = y

    fig = plt.figure()
    titles = ['x', 'y']
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        ax.set_xlabel('x [10^(-6) m]')
        ax.set_ylabel('Bin height')
        bin_centres, y, a, b = histogram_and_fit_quadratic(
            xy[:, i]
            )
        ax.set_title('k_{} = {:.2e}J/m^2'.format(
            titles[i], 8.09*10.**(-9.)*a))
        ax.scatter(bin_centres, y)
        x_model = np.linspace(min(bin_centres), max(bin_centres), 100)
        ax.plot(x_model, a*x_model**2. - b)
    fig.tight_layout()
    plt.show()

    return None