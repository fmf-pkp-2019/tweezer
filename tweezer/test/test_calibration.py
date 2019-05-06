"""Unit tests for the calibration module"""

import unittest

import numpy as np
import tweezer.calibration as cal
import tweezer.calibration_generate_data as gen

class TestCalibration(unittest.TestCase):
    
    def setUp(self):
        self.kx = 1.e-6
        self.ky = 2.e-6
        self.phi = 0.4
        self.expected_result = np.array([tuple([self.kx, self.ky]), self.phi])
        
    def test_center_and_rotate(self):
        trajectory = gen.generate([self.kx, self.ky], phi = self.phi, drifting = False)
        result = cal.center_and_rotate(trajectory[:, 0], trajectory[:, 1])
        self.assertTrue(np.allclose(-result[1], self.expected_result[1], atol = 0.01))
    
    def test_moving_average(self):
        trajectory = gen.generate([self.kx, self.ky], phi = self.phi)
        t = gen.generate_time()
        x, _, _ = cal.subtract_moving_average(t, trajectory[:, 0], 1)
        y, _, _ = cal.subtract_moving_average(t, trajectory[:, 1], 1)
        self.assertTrue(np.allclose(np.mean(x), 0, atol = 1e-4) and np.allclose(np.mean(y), 0, atol = 1e-4))
        
    def test_calibrate(self):
        trajectory = gen.generate([self.kx, self.ky], phi = self.phi)
        t = gen.generate_time()
        result = cal.calibrate(t, trajectory)
        self.assertTrue(np.allclose(result[0], self.expected_result[0], atol=1e-6) and
                np.allclose(-result[1], self.expected_result[1], atol=0.1))
        

if __name__ == "__main__":
    unittest.main()
    
    
