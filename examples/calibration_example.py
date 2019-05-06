import tweezer.calibration as cal
import tweezer.calibration_generate_data as gen
import tweezer.plotting as plt

# Parameters for drawing positions
k = (1.e-06, 2.e-6)
phi = 0.4
center = (6., 7.)

# Drawing positions
data = gen.generate(k, phi=phi, center=center)
time = gen.generate_time()
averaging_time = 0.1

# Example of using calibration.calibrate
k_estimated, phi_estimated, center_estimated = cal.calibrate(time, data, averaging_time)
print(k_estimated, phi_estimated)

# Example of using calibarion.potential
positions, values, phi = cal.potential(time, data, averaging_time)

# Example of using trajectory plot
plt.trajectory_plot(time, data, averaging_time)

# Example of using calibration plots
plt.calibration_plots(time, data, averaging_time)

# Example of using potential plot
plt.potential_plot(time, data, averaging_time)