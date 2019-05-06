"""
Tracks particles.
Opens file, tracks particles, links particles, gets trap data and saves
everything into .dat file.
"""

import unittest
import pims
import trackpy as tp
from tweezer import TWV_Reader

def save_tracked_data(filename, frames, trajectories, times, laser_powers, traps):
    """
    Converts all data to a (.dat) file.
    Important NOTE:
    The format will change in the future. Particle data and trap data will go into separate files,
    and additional trap metadata will be saved.
    
    Inputs:
     - filename: Output filename.
     - frames: TWV_Reader or other pims object. __len__ needs to be defined.
     - trajectories: Pandas DataFrame with all the particles. Same structure as returned by TrackPy.
     - times: List of frame times.
     - laser powers: List of laser powers for every frame.
     - traps: List of traps data for every frame. traps[frame number][trap number][power/position_x/position_y]
     
    Output format, tab separated:
     - time, laser power, trap_1_power, trap_1_x, trap_1, y,
         the same for traps 2-4, particle_1_x, particle_1_y,
         the same for all particles
    If a particle is missing from a frame, empty string ('') is placed
    instead of coordinates.
    """
    max_particles = int(round(trajectories.max()['particle']))
    with open(filename, 'w') as f:
        for i in range(len(frames)):
            tmp = ''
            tmp += str(times[i]) + '\t'
            tmp += str(laser_powers[i]) + '\t'
            for j in range(4): # for j in traps
                for k in range(3): # for k in power/x/y of a trap
                    tmp += str(traps[j][i][k]) + '\t'
            for j in range(max_particles+1):
                tmp_particle = trajectories.loc[
                    trajectories['particle'] == j].loc[
                        trajectories['frame'] == i]
                # find the particle j on frame i
                if tmp_particle.empty:
                    tmp += '\t\t'
                    # if no such particle exists, write two tabs
                else:
                    tmp += str(tmp_particle.iloc[0]['x']) + '\t'
                    tmp += str(tmp_particle.iloc[0]['y']) + '\t'
                    # else write the particles position
            tmp += '\n'
            f.write(tmp)

class Test(unittest.TestCase):
    """
    Also tests TWV_Reader module.
    """

    def setUp(self):
        pass

    def test_everything(self):
        """
        End to end test.
        Test the whole tracking pipeline from input file to particle and trap
        positions in the output file.
        """
        filename = "../examples/data/test_example.twv"
        frames = pims.open(filename)
        times, laser_powers, traps = frames.get_all_tweezer_positions()
        features = tp.batch(frames, 25, minmass=1000, invert=False)
        tracks = tp.link_df(features, 15, memory=10)
        save_tracked_data(filename[:-4] + '_out.dat', frames, tracks, times, laser_powers, traps)
        with open(filename[:-4] + '_out.dat', 'r') as calculated_file:
            with open(filename[:-4] + '_expected.dat', 'r') as expected_file:
                for calculated, expected in zip(calculated_file, expected_file):
                    self.assertEqual(calculated, expected)

def example(filename):
    """
    Example usecase from input file to particle and trap positions in .dat file.
    """
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

if __name__ == "__main__" and 0:
    example("../examples/data/test_example.twv")
					
if __name__ == "__main__":
    unittest.main()
