"""
Example usecase from input file to particle and trap positions in .dat file.
"""

import pims
import trackpy as tp
from tweezer import TWV_Reader
from tweezer.tracking import save_tracked_data

filename = "data/test_example.twv"

frames = pims.open(filename)
# Open file with pims. Works with many file extensions.
# This example assumes .twv file.

# metadata = frames.get_all_metadata()
# Optional access to additional metadata.

times, laser_powers, traps = frames.get_all_tweezer_positions()
# Obtain frame times, laser power at each frame time and
# traps powers and positions at each frame.

features = tp.batch(frames, 25, minmass=1000, invert=False)
# Obtain features (particle positions) using trackpy's batch function.
# It is verbose.
# The 25 in arguments is diameter. It be odd number.
# It is recommended to obtain parameters using GUI.

tracks = tp.link_df(features, 15, memory=10)
# Joins particles positions to tracks (connects them in time).
# See trackpy documentation for parameters.

save_tracked_data(filename[:-4] + '_out.dat', frames, tracks, times, laser_powers, traps)
# Save data in a format readable by other scripts.
