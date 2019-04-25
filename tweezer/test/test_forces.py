import unittest
import random
import os
import numpy as np

import tweezer.synth_active_trajectory as sat
import tweezer.force_calc as forcecalc
import tweezer.plotting as plt

class TestForce(unittest.TestCase):
    """Unit testing for scripts that simulate the Brownian motion of particle in trap
    and calculate the forces on it.
    """
    def setUp(self):
        random.seed(123)

    def test_simulation_calc(self):

        precalculated_coeffs = (0.0082505622,0.0082126888)
        precalculated_means = (0.10855325,0.0509324)
        
        kx_estimate,ky_estimate = sat.SAT2("unit_test.dat",1000,0.005, 2.5e-6, 0.5e-6, 2, 1, 1e-6, 1e-6, 0.5e-6, 9.7e-4, 300, 1)
        
        np.testing.assert_approx_equal(kx_estimate,precalculated_coeffs[0],6)  # Test to 6 significant digits
        np.testing.assert_approx_equal(ky_estimate,precalculated_coeffs[1],6)
        
        time, traps, trajectories = plt.read_file("unit_test.dat", 1)
        _, means = forcecalc.force_calculation(time, trajectories[:, 0:2], traps[:, 0:2], (2.5e-6,0.5e-6), 300)
        self.assertTrue(np.allclose(means, precalculated_means, rtol=1e-05, atol=1e-08))
        
    def tearDown(self):
        os.remove("unit_test.dat")  # Cleaning up 

if __name__ == "__main__":
    unittest.main()
