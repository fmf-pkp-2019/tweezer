"""Unit tests for the force calculation module"""

import sys
sys.path.append('../')

import unittest
import random
import math
import os
import numpy as np

import synth_active_trajectory as sat
import force_calc as forcecalc
import plotting as plt
import calibration_generate_data as generate

class TestForce(unittest.TestCase):
    """Tests scripts which simulate the Brownian motion of particle in trap and calculate the forces on it.
    """
    
    def setUp(self):
        random.seed(123)
        num_points = 10**3

        pos_1 = (-60.,12.)
        pos_2 = (5.,30.)
        self.k = (2.5e-6,0.5e-6)

        self.expected_coeffs = (0.0082505622,0.0082126888)
        self.expected_means_calc = (0.10855325,0.0509324)        
        self.expected_means_axis = (0.69330303,0.42569363)
        self.expected_means_displacements = (0.3993498,0.85102993,0.32364116,0.51692861)
        self.expected_distance = 67.446274915669
        self.expected_sigmas = (0.20100778,0.12462615,0.14673569,0.1178954)

        data_1 = generate.generate(self.k,273,random.uniform(0,2*math.pi),(pos_1[0]+random.uniform(0,1),pos_1[1]+random.uniform(0,1)),num_points)
        data_2 = generate.generate(self.k,273,random.uniform(0,2*math.pi),(pos_2[0]+random.uniform(-1,0),pos_2[1]+random.uniform(-1,0)),num_points)
        #random displacements represent the inter-particle interaction

        position = np.hstack((pos_1,pos_2))
        self.times = np.zeros((num_points,1))
        self.positions = np.repeat(position[None, :], num_points, axis=0)
        self.trajectories = np.hstack((data_1,data_2))

    def test_simulation_calc(self):
        random.seed(123)
        
        kx_estimate,ky_estimate = sat.SAT2("test_simulation.dat",1000,0.005, self.k[0], self.k[1], 2, 1, 1e-6, 1e-6, 0.5e-6, 9.7e-4, 300, 1)
        
        np.testing.assert_approx_equal(kx_estimate,self.expected_coeffs[0],6)  # Test to 6 significant digits
        np.testing.assert_approx_equal(ky_estimate,self.expected_coeffs[1],6)
        
        time, traps, trajectories = plt.read_file("test_simulation.dat", 1)
        _, means = forcecalc.force_calculation(time, trajectories[:, 0:2], traps[:, 0:2], (2.5e-6,0.5e-6), 300)

        self.assertTrue(np.allclose(means, self.expected_means_calc, rtol=1e-05, atol=1e-08))
    
        os.remove("test_simulation.dat")

    def test_force_axis(self):
        _,means,distance = forcecalc.force_calculation_axis(self.times, self.trajectories, self.positions, self.k, self.k, temp=293)

        np.testing.assert_approx_equal(distance,self.expected_distance,6)
        self.assertTrue(np.allclose(means, self.expected_means_axis, rtol=1e-05, atol=1e-08))

    def test_force_displacements(self):
        means,sigmas = forcecalc.sigma_calculation(self.trajectories, self.positions)

        self.assertTrue(np.allclose(means, self.expected_means_displacements, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(sigmas, self.expected_sigmas, rtol=1e-05, atol=1e-08))

if __name__ == "__main__":
    unittest.main()
