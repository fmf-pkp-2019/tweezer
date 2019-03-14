# %%
import numpy as np
import scipy.optimize
import scipy.constants

import matplotlib.pyplot as plt


KB = scipy.constants.Boltzmann


def center_and_rotate(xdata, ydata):
    """
    Sets the average position as the origin,
    makes the major axis of the ellipse lie on the x-axis.

    Args:
        xdata: Array of x-positions.
        ydata: Array of y-positions.

    Returns:
        Array of centered and rotated x-positions,
        array of centered and rotated y-positions,
        angle of rotation in the anticlockwise direction.
    """

    if len(xdata) != len(ydata):
        raise ValueError("Unclear number of points.")

    def center(data):
        return data - np.mean(data)

    def rotate(x, y):
        n = len(x)

        cov = np.cov(x, y)
        var, vec = np.linalg.eigh(cov)
        phi = -np.arctan(vec[:, 1][1]/vec[:, 1][0])

        def rotate_one(phi, x, y):
            x_rotated = np.cos(phi)*x - np.sin(phi)*y
            y_rotated = np.sin(phi)*x + np.cos(phi)*y
            return x_rotated, y_rotated

        x_rotated, y_rotated = (np.zeros(n), np.zeros(n))
        for i in range(n):
            x_rotated[i], y_rotated[i] = rotate_one(
                phi, x[i], y[i]
            )
        return x_rotated, y_rotated, phi, var[::-1]

    return rotate(center(xdata), center(ydata))


def calibrate(time, xdata, ydata, temp=293):

    x = subtract_linear_drift(time, xdata)
    y = subtract_linear_drift(time, ydata)

    x, y, phi, var = center_and_rotate(x, y)
    k = KB*temp/var*1e12

    return tuple(k), phi


def plot(time, xdata, ydata, temp=293):

    x = subtract_linear_drift(time, xdata)
    y = subtract_linear_drift(time, ydata)
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
        ax.set_title('k{} = {:.2e}J/m^2'.format(
            titles[i], k[i]))
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
        ax.scatter(bin_centres, hist, s=4)
        x_model = np.linspace(min(bin_centres), max(bin_centres), 100)
        prefactor = 1./np.sqrt(2.*np.pi*var[i])
        ax.plot(x_model, prefactor*np.exp(-x_model**2./(2.*var[i])))
    fig.tight_layout()
    plt.show()

    return None
