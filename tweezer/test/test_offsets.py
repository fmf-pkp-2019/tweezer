""Unit tests for the offset calibration module"""

import sys
sys.path.append('../')

import unittest
import os
import random
import numpy as np

import offset
import calibration_generate_data as generate

class TestOffsetFourCorner(unittest.TestCase):

    def setUp(self):
        num_points = 10**3
        k = (1.0,1.0)
        corners = np.array(([-100.,100.],[100.,100.],[-100.,-100.],[100.,-100.]))

        p1 = np.array([-1.08952728e+02,9.17437340e+01,1.93793530e-02,-3.39998759e-01])
        p2 = np.array([1.08023978e+02,9.07630730e+01,2.77299490e-01,1.08996929e-01])
        p3 = np.array([-9.29582680e+01,-1.06806752e+02,-1.20670450e-01,1.12193934e-01])
        p4 = np.array([9.49032670e+01,-1.09966589e+02,7.74474840e-02,-3.80171711e-01])

        self.expected_pts = np.vstack((p1,p2,p3,p4))        
        self.calibration_expected_result = (0.070487528, -0.121645459)

        random.seed(123)

        for n in range(1,5):
            rand = (random.uniform(-10.0,10.0),random.uniform(-10.0,10.0))
            data = generate.generate(k,273,0,(corners[n-1,0]+rand[0]+random.uniform(-1.,1.),corners[n-1,1]+rand[1]+random.uniform(-1.,1.)),num_points)

            fout = open("offset_%s.dat" % str(n),"w")
            for i in range(0,num_points):
                fout.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t\n" 
                % (float(i),1.,corners[n-1,0]+rand[0],corners[n-1,1]+rand[1],1.,0,0,-1,0,0,-1,0,0,-1,data[i,0],data[i,1]))
            fout.close()

    def test_four_corner_offsets(self):

        pts = offset.four_corner_offsets("offset_1.dat","offset_2.dat","offset_3.dat","offset_4.dat")
        self.assertTrue(np.allclose(self.expected_pts, pts, atol=1e-6))

    def test_four_corner_calibration(self):

        result = offset.four_corner_calibration(0.,0.,self.expected_pts)
        self.assertTrue(np.allclose(result, self.calibration_expected_result, atol=1e-6))

    def tearDown(self):
        os.remove("offset_1.dat")
        os.remove("offset_2.dat")
        os.remove("offset_3.dat")
        os.remove("offset_4.dat")
        
if __name__ == "__main__":
    unittest.main()
