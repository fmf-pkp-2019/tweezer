# %%
import numpy as np
import scipy.constants

KB = scipy.constants.Boltzmann


def draw(ks, temp, number_of_points):
    """
    k .. tuple (kx, ky)
    """

    np.random.seed(seed=0)
    u1 = np.random.rand(number_of_points)
    u2 = np.random.rand(number_of_points)

    z1 = np.sqrt(-2. * np.log(u1)) * np.cos(2.*np.pi * u2)
    z2 = np.sqrt(-2. * np.log(u1)) * np.sin(2.*np.pi * u2)

    variance_x = KB*temp/ks[0]*1e12
    variance_y = KB*temp/ks[1]*1e12

    x = np.sqrt(variance_x) * z1
    y = np.sqrt(variance_y) * z2

    return x, y


def rotate(x, y, phi):
    def rotate_one(phi, x, y):
        x_rotated = np.cos(phi)*x - np.sin(phi)*y
        y_rotated = np.sin(phi)*x + np.cos(phi)*y
        return x_rotated, y_rotated

    xy_rotated = np.zeros((len(x), 2))
    for i in range(len(xy_rotated)):
        xy_rotated[i] = rotate_one(
            phi, x[i], y[i]
        )
    return xy_rotated[:, 0], xy_rotated[:, 1]


# def drift(x, y, *args):


def generate(ks, temp=273, phi=0, number_of_points=10**4):
    x, y = draw(ks, temp, number_of_points)
    x, y = rotate(x, y, phi)
    # x, y = drift(x, y, *args)
    return x, y
