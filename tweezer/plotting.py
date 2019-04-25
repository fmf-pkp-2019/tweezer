import numpy as np
import matplotlib.pyplot as plt
import tweezer.calibration as cal

def read_file(path, no_of_particles):
    """Unpacks a dat file.

    Parameters
    ----------
    path : string
        location and the name of dat file
    no_of_particles : int
        number of particles in interest

    Returns
    -------
    time: array_like
        timestamps
    traps: ndarray_like
        location and strength of 4 traps
    trajectories: 2D array
        x and y trajectories of no_of_particles

    Examples
    --------
    TODO
    """
    raw_data = open(path, "r")
    columns = int(14+2*no_of_particles)
    data = np.zeros((100000, columns))
    rows = 0
    # Read file by lines
    for line in raw_data.readlines():
        # If data is missing (double tab), replace it with nan
        line = line.replace('\t\t', '\tnan')
        data[rows,:] = (line.split('\t'))[:columns]
        rows += 1
    data = data[:rows, :].astype(np.float)
    print('Shape of initial data: ', data.shape)
    # Check data how many nans it contains in trajectories
    for i in range(rows-1, 0, -1):
        check_row = np.isnan(data[i, 14:])
        # Delete row, if it contains nan
        if(np.sum(check_row) > 0):
            data = np.delete(data, i, 0)
    print('Shape of cropped data: ', data.shape)
    return data[:, 0], data[:, 2:14], data[:, 14:columns]

def trajectory_plot(time, data, averaging_time=1.):
    """
    TODO

    """
    data = np.array(data)
    x, x_average, _ = cal.subtract_moving_average(time, data[:, 0], averaging_time)
    y, y_average, time = cal.subtract_moving_average(time, data[:, 1], averaging_time)
    
    trajectory = np.stack((x, y), axis=1)
    trajectory_averaged = np.stack((x_average, y_average), axis=1)

    fig = plt.figure()
    titles = ['x', 'y']
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        ax.set_title('Trajectory of a trapped particle in {} direction'.format(titles[i]))
        ax.grid(True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Direction {} '.format(titles[i]) + r'[$\mu m$]')
        ax.scatter(time, trajectory[:, i] + trajectory_averaged[:, i], s=4, label = r'original')
        ax.scatter(time, trajectory_averaged[:, i], s=4, label = r'averaged')
        ax.legend(loc = 'best')
    fig.tight_layout()
    plt.show()
    return None

def calibration_plots(time, data, averaging_time=1., temp=293.):
    """
    TODO (Taken from the calibration.py)

    """
    data = np.array(data)
    x = cal.subtract_moving_average(time, data[:, 0], averaging_time)[0]
    y = cal.subtract_moving_average(time, data[:, 1], averaging_time)[0]
    trajectory, phi, var = cal.center_and_rotate(x, y)
    k = cal.KB*temp/var*1e12

    def scatter_plot(data, trajectory, phi):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Original data')
        ax1.grid(True)
        ax1.set_xlabel('Direction x ' + r'[$\mu m$]')
        ax1.set_ylabel('Direction y ' + r'[$\mu m$]')
        ax1.scatter(data[:, 0], data[:, 1], s=4)
        ax1.set_aspect('equal')
        ax2.set_title('Centered data, phi = {:.2f} rad'.format(phi,))
        ax2.grid(True)
        ax2.set_xlabel('Direction x ' + r'[$\mu m$]')
        ax2.set_ylabel('Direction y ' + r'[$\mu m$]')
        ax2.scatter(trajectory[:, 0], trajectory[:, 1], s=4)
        ax2.set_aspect('equal')
        fig.tight_layout()
        plt.show()
        return None

    def histogram_plot(trajectory, var):
        fig = plt.figure()
        titles = ['x', 'y']
        for i in range(2):
            ax = fig.add_subplot(1, 2, i+1)
            ax.set_xlabel(('Direction {} ' + r'[$\mu m$]').format(titles[i]))
            ax.set_ylabel('Bin height')
            hist, bin_edges = np.histogram(trajectory[:, i], bins=int(np.sqrt(len(trajectory[:, i]))), density=True)
            ax.set_title('k_{} = {:.2e}J/m^2'.format(titles[i], k[i]))
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
            ax.scatter(bin_centres, hist, s=4)
            x_model = np.linspace(min(bin_centres), max(bin_centres), 100)
            prefactor = 1./np.sqrt(2.*np.pi*var[i])
            ax.plot(x_model, prefactor*np.exp(-x_model**2./(2.*var[i])),label = r'fit')
            ax.legend(loc = 'best')
        fig.tight_layout()
        plt.show()
        return None

    scatter_plot(data, trajectory, phi)
    histogram_plot(trajectory, var)
    return None

def potential_plot(time, data, averaging_time=1., temp=293.):
    """
    TODO (Taken from the calibration.py)

    """
    positions, potential_values, _ = cal.potential(time, data, averaging_time= averaging_time, temp=temp)

    fig = plt.figure()
    titles = ['x', 'y']
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1)
        ax.set_title('Shape of a potential in {} direction'.format(titles[i]))
        ax.set_xlabel(('Direction {} ' + r'[$\mu m$]').format(titles[i]))
        ax.set_ylabel('Potential [kT]')
        ax.scatter(positions[i], potential_values[i], s=4)
    fig.tight_layout()
    plt.show()
    return None

def force_plot(time, forces):
    """
    TODO (Taken from the force_calc.py)

    For now creates a basic F(t) plot.
    Parameters
    ----------
    time : list of floats
        times of recorded points
    forces : ndarray of floatd
        two-column array of forces on trapped bead in x- and y-directions
    """        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Radial gradient forces on trapped particle')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('F [pN]')
    ax.plot(time, forces[:, 0]*1e6, label = r'$F_x$')
    ax.plot(time, forces[:, 1]*1e6, label = r'$F_y$')
    ax.plot(time, np.sqrt(forces[:, 0]**2 + forces[:, 1]**2)*1e6, label = r'$F_{sum}$')
    ax.grid(True)
    ax.legend(loc = 'best')
    fig.tight_layout()
    plt.show()
    return None