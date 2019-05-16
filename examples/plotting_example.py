import os
import numpy as np
import tweezer.plotting as plt
import tweezer.force_calc as forcecalc
import tweezer.calibration as cal

# Example of unpacking the data, and plotting all plot functios
# Get data from drive (under Vzorci/1 static particle) and save it in
# examples folder

# script_dir = os.path.dirname(__file__) # absolute dir the script is in
file_name = 'test.dat'  # name of the dat file
# path = os.path.join(script_dir, file_name)
path = file_name
particles = 1

time, traps, trajectories = plt.read_file(path, particles)
averaging_time = 0.1

for i in range(particles):
    data = trajectories[:, i*2:(i+1)*2]
    trap = traps[:, i*3:(i+1)*3]
    # Example of using trajectory plot
    plt.trajectory_plot(time, data, averaging_time)
    # Example of using calibration plots
    plt.calibration_plots(time, data, averaging_time)
    # Example of using potential plot
    plt.potential_plot(time, data, averaging_time)
    # Example of using potential plot with polynomial fit
    plt.potential_polynomial_fit_plot(time, data, order=4)
    # Calculate forces
    k_estimated, phi_estimated, center_estimated = cal.calibrate(
        time, data, averaging_time)
    f, m = forcecalc.force_calculation(time, data, trap, k_estimated, 300)
    # Plot force
    plt.force_plot(time, f)
