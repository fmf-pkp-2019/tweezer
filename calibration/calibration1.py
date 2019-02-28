# %%
import numpy as np
import scipy.optimize
import scipy.constants

import matplotlib.pyplot as plt


def subtract_linear_drift(time, data):
    """
    Subtracts the linear drift from positions.

    Args:
        time: Array of times.
        data: Array of positions in one dimension.

    Returns:
        Array of positions with the linear drift subtracted.
    """

    def linear(x, k, n):
        return k*x+n

    (k, n), _ = scipy.optimize.curve_fit(linear, time, data)

    return data - linear(time, k, n)


def center_and_rotate(xdata, ydata):
    """
    Sets the average position as the origin,
    makes the major axis of the ellipse lie on the x-axis.

    Args:
        xdata: Array of x-positions.
        ydata: Array of y-positions.

    Returns:
        Array of centered and rotated x-positions,
        array of centered and rotated y-positions.
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

    return xy_rotated[:, 0], xy_rotated[:, 1]


def histogram_and_fit_gaussian(data):
    """
    Calibrates by histogramming position deviations
    (into int(np.ceil(np.sqrt(len(data)))) bins)
    by fitting a gaussian function
    y = prefactor e^(-x^2/(2 variance)).

    Args:
        data: Array of position deviations in one dimension.

    Returns:
        Centres of histogram bins,
        heights of bins,
        prefactor,
        variance.
    """

    heights, bin_edges = np.histogram(
        data, bins=int(np.ceil(np.sqrt(len(data)))),
        density=True
    )
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

    def gaussian(x, prefactor, variance):
        return prefactor*np.exp(-x**2./(2.*variance))

    (prefactor, variance), _ = scipy.optimize.curve_fit(
        gaussian, bin_centres, heights,
        p0=[1., np.var(data)]
    )

    return bin_centres, heights, prefactor, variance


def calibrate1(time, xdata, ydata, temp=293):
    """
    Calibrates a tweezer using histogram_and_fit_gaussian.

    Args:
        xdata: Array of x-positions.
        ydata: Array of y-positions.

    Returns:
        (k_x, k_y).
    """

    x = subtract_linear_drift(time, xdata)
    y = subtract_linear_drift(time, ydata)
    x, y = center_and_rotate(x, y)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original data')
    ax1.set_xlabel('x [10^(-6) m]')
    ax1.set_ylabel('y [10^(-6) m]')
    ax1.scatter(xdata, ydata, s=4)
    ax1.set_aspect('equal')
    ax2.set_title('Centered data')
    ax2.set_xlabel('x [10^(-6) m]')
    ax2.set_ylabel('y [10^(-6) m]')
    ax2.scatter(x, y, s=4)
    ax2.set_aspect('equal')
    fig.tight_layout()
    plt.show()

    xy = np.zeros((len(xdata), 2))
    xy[:, 0] = x
    xy[:, 1] = y

    ks = []

    fig = plt.figure()
    titles = ['x', 'y']
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        ax.set_xlabel('x [10^(-6) m]')
        ax.set_ylabel('Bin height')
        bin_centres, heights, prefactor, variance = histogram_and_fit_gaussian(
            xy[:, i]
        )
        k = scipy.constants.Boltzmann*temp/variance*1e12
        ks.append(k)
        ax.set_title('k_{} = {:.2e}J/m^2'.format(
            titles[i], k))
        ax.scatter(bin_centres, heights, s=4)
        x_model = np.linspace(min(bin_centres), max(bin_centres), 100)
        ax.plot(x_model, prefactor*np.exp(-x_model**2./(2.*variance)))
    fig.tight_layout()
    plt.show()

    return tuple(ks)
