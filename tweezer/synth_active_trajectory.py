import math
import random
import statistics

import numpy as np
import scipy.constants as constants

def SAT1(file_name, num_points, dt, trap_k, trap_frequency, trap_amplitude, bead_radius, eta):
    """Simulates the Brownian motion of a colloidal bead trapped in an optical trap oscillating in the x direction.
    
    Originally implemented in MATLAB by Natan Osterman, advised by Andrej Vilfan, 23.3.2012.
    numPoints rows and 4 columns

    Parameters
    ----------
    file_name : string
        generated data will be stored here
    num_points : int
        # of data points to generate 
    dt : float
        time interval between two consecutive points [s]
    trap_k : float
        trap stiffness [N/m]
    trap_frequency : float
        trap oscillation frequency [Hz]
    trap_amplitude : float
        amplitude of oscillation [m]
    bead_radius : float
        radius of trapped particle [m]
    eta : float
        viscosity of medium [Pa s]

    Note
    ----
    Position values in output file are in micrometers!
    
    Returns
    -------
    kx_calculated : float
        estimated trap stiffness in x-direction
    ky_calculated : float
        estimated trap stiffness in y-direction
    file
        file called file_name with 4 columns: point #, time, x-, y-coordinates of bead
    """
    
    print("Calculating ...")
    fout = open(file_name, "w")

    kBT = constants.Boltzmann*300  # assumption: T=300K
    a = bead_radius
    x = np.array([0.,0.])
    trap_x = np.array([0.,0.])
    noise = np.array([0.,0.])
    dx = np.array([0.,0.])
    dt_internal=0.0002  # internal time step used for simulation [s]

    poz = np.zeros([num_points,2])  # initial position of bead [x y] in meters
    trap_poz = np.zeros([num_points,2])  # initial trap position
    time = np.zeros([num_points])

    i=0
    t=0
    last_sample_interval=dt+1e-10

    while i < num_points:
        if last_sample_interval > dt:
            last_sample_interval -= dt
            i += 1
            fout.write("%d %3.3f %3.3f %3.3f\r\n" % (i,t,x[0]*1e6,x[1]*1e6))
            poz[i-1,:] = x
            trap_poz[i-1,:] = trap_x
            time[i-1] = t
        t += dt_internal
        last_sample_interval += dt_internal
        noise = [(2*random.random()-1)*math.sqrt(3),(2*random.random()-1)*math.sqrt(3)]
        trap_x = [trap_amplitude*math.sin(2*math.pi*trap_frequency*t),0]
        dx = ((-trap_k*dt_internal)/(6*math.pi*eta*a))*(np.array(x)-np.array(trap_x))+(math.sqrt(2*kBT/(6*math.pi*eta*a))*math.sqrt(dt_internal))*np.array(noise)
        x += dx
    
    fout.close()

    k_calculated = np.zeros([num_points,2])
    for i in range(0,num_points):
        k_calculated[i,0] = poz[i,0]*poz[i,0]
        k_calculated[i,1] = poz[i,1]*poz[i,1]

    kx_calculated=kBT/statistics.mean(k_calculated[:,0])
    ky_calculated=kBT/statistics.mean(k_calculated[:,1])

    return kx_calculated*1e6, ky_calculated*1e6

def SAT2(file_name, num_points, dt, trap_kx, trap_ky, trap_xfreq, trap_yfreq, trap_xamp, trap_yamp, bead_radius, eta, temp=293, motion_type=1):
    """Simulates the Brownian motion of a colloidal bead trapped in an optical trap oscillating in x and y directions.

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

    Note
    ----
    Position values in output file are in micrometers!
    
    Returns
    -------
    kx_estimate : float
        estimated trap stiffness in x-direction
    ky_estimate : float
        estimated trap stiffness in y-direction
    file
        file called file_name with 6 columns: point #,time,x-,y-coords of bead,x-,y-coords of trap

    Raises
    ------
    ValueError
        if times between consecutive output points are less than timestep of simulation
    """
    dt_internal=0.0001   #   internal time step used for simulation [s]

    if (dt <= dt_internal):
        raise ValueError("dt must be longer than time step of simulation")

    print("\nCalculating ...")
    fout = open(file_name, "w")

    kBT = constants.Boltzmann*temp   #   assumption: T=300K
    a = bead_radius
    x = np.array([0.,0.])
    trap_x=np.array([0.,0.])
    trap_k = np.array([trap_kx,trap_ky])
    noise = np.array([0.,0.])
    dx = np.array([0.,0.])
    
    poz = np.zeros([num_points,2])   #   initial position of bead [x y] in meters
    trap_poz = np.zeros([num_points,2])   #   initial position of trap
    time = np.zeros([num_points])

    i=0
    t=0
    last_sample_interval=dt+1e-10

    while i < num_points:
        if last_sample_interval > dt:
            last_sample_interval -= dt
            i += 1
            fout.write("%3.5f\t\t\t%3.3f\t%3.3f\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t%3.4f\t%3.4f\t\n" % (t,trap_x[0]*1e6,trap_x[1]*1e6,x[0]*1e6,x[1]*1e6))
            poz[i-1,:] = x
            trap_poz[i-1,:] = trap_x
            time[i-1] = t
        t += dt_internal
        last_sample_interval += dt_internal
        noise[0] = (2*random.random()-1)*math.sqrt(3)
        noise[1] = (2*random.random()-1)*math.sqrt(3)

        if (motion_type == 1):
            trap_x[0] = trap_xamp*math.sin(2*math.pi*trap_xfreq*t)
            trap_x[1] = trap_yamp*math.cos(2*math.pi*trap_yfreq*t)
        elif (motion_type == 2):
            trap_x[0] = trap_xamp*t
            trap_x[1] = trap_yamp*t

        dx = (dt_internal/(6*math.pi*eta*a))*(np.array(trap_k)*(np.array(trap_x)-np.array(x))) + (math.sqrt(2*kBT/(6*math.pi*eta*a))*math.sqrt(dt_internal))*np.array(noise)
        x += dx
        
    fout.close()

    k_estimate = np.zeros([num_points,2])
    for i in range(0,num_points):
        k_estimate[i,0] = poz[i,0]*poz[i,0]
        k_estimate[i,1] = poz[i,1]*poz[i,1]

    kx_estimate=kBT/statistics.mean(k_estimate[:,0])*1e6
    ky_estimate=kBT/statistics.mean(k_estimate[:,1])*1e6

    return kx_estimate, ky_estimate
