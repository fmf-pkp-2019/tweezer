import tweezer.force_calc as forcecalc
import tweezer.synth_active_trajectory as sat
import tweezer.plotting as plt

# Parameters for drawing positions
k_values = (2.5e-6,0.5e-6)

# The program first generates simulated motion of a bead moving in an optical trap
# and saves it into a file.
sat.SAT2("test.dat",1000,0.005, k_values[0], k_values[1], 2, 1, 1e-6, 1e-6, 0.5e-6, 9.7e-4, 300, 1)

# The file is read back.
time, traps, trajectories = plt.read_file("test.dat", 1)

# Calculate forces
f,m = forcecalc.force_calculation(time, trajectories[:, 0:2], traps[:, 0:2], k_values, 300)

# Plot force
plt.force_plot(time,f)
