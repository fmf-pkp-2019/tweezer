import sys
sys.path.append('../')

import tweezer.force_calc as forcecalc
import tweezer.synth_active_trajectory as sat
import tweezer.plotting as plt
import tweezer.calibration_generate_data as gen

# Parameters for simulation
k_values = (2.5e-6,0.5e-6)
k_values_2 = (1e-6,1e-6)
viscosity = 9.7e-4
temperature = 300
num_of_points = 10**3

# Example of generating simulated motion of a bead moving in an optical trap
# and saving it into a file
sat.SAT2("test.dat",num_of_points,0.005, k_values[0], k_values[1], 2, 1, 1e-6, 1e-6, 0.5e-6, viscosity, temperature, 1)

# Reading the file back
time, traps, trajectories = plt.read_file("test.dat", 1)

# Calculating forces
f,m = forcecalc.force_calculation(time, trajectories[:, 0:2], traps[:, 0:2], k_values, temperature)

# Plotting forces
plt.force_plot(time,f)

# Example of calculating force of interaction between a pair of beads and their displacement
time,traps,trajectories = plt.read_file("ES0_5_out.dat",2)

f,m,r = forcecalc.force_calculation_axis(time,trajectories,traps,k_values,k_values_2,temperature)

plt.force_plot(time,f)

d,s = forcecalc.sigma_calculation(trajectories,traps)
