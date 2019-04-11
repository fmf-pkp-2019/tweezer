import tweezer.force_calc as forcecalc
import tweezer.synth_active_trajectory as sat

"""A minimal workflow example.
The program first generates simulated motion of a bead moving in an optical trap and saves it into a file.
Then, the file is read back, forces are calculated and plotted. 

Information on function parameters (copied from the corresponding docstrings):

SAT2
    Parameters
    ----------
    file_name : string
        generated data will be stored here
    num_points : int
        # of data points to generate 
    dt : float
        time interval between two consecutive points [s]
    trap_kx : float
        trap stiffness in x-direction [N/m]
    trap_ky : float
        trap stiffness in y-direction [N/m]
    trap_xfreq : float
        trap oscillation frequency in x-direction [Hz]
    trap_yfreq : float
        trap oscillation frequency in y-direciton [Hz]
    trap_xamp : float
        amplitude of oscillation in x-direction [m]
    trap_yamp : float
        amplitude of oscillation in y-direction [m]
    bead_radius : float
        radius of trapped particle [m]
    eta : float
        viscosity of medium [Pa s]
    temp : float
        system temperature [K]
    motion_type : int
        governs trap motions and can take values 1 or 2.
        1: sinusoidal motion in x,y (default)
        2: linear motion in x,y; in this case, frequency parameters are ignored and the amplitudes become velocities in [m/s]

read_input
    Parameters
    ----------
    filename : str
        name of file to open

force_calculation
    Parameters
    ----------
        time : list of float
            times of 
        pos_x : list of float
            x-positions of trapped bead
        pos_y : list of float
            y-positions of trapped bead
        trap_pos_x : list of float
            x-positions of trap
        trap_pos_y : list of float
            y-positions of trap
        ks : tuple of float
            trap stiffnesses in x- and y-directions [N/m]
        temp : float
            system temperature [K]

force_plotting
    Parameters
    ----------
    time : list of float
        times of recorded points
    forces : np.array of float
        two-column array of forces on trapped bead in x- and y-directions
"""

k_values = (2.5e-6,0.5e-6)

sat.SAT2("test.dat",1000,0.005, k_values[0], k_values[1], 2, 1, 1e-6, 1e-6, 0.5e-6, 9.7e-4, 300, 1)
values = forcecalc.read_input("test.dat")

# The calculated values are recorded in "f".

times = values[:,0]

f,m = forcecalc.force_calculation(times, values[:,1], values[:,2], values[:,3], values[:,4], k_values, 300)

forcecalc.force_plotting(times,f)
