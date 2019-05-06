import unittest
import pims
import trackpy as tp
from tweezer import TWV_Reader
from tweezer.tracking import save_tracked_data

class TestTracking(unittest.TestCase):
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
        filename = "../../examples/data/test_example.twv"
        frames = pims.open(filename)
        times, laser_powers, traps = frames.get_all_tweezer_positions()
        features = tp.batch(frames, 25, minmass=1000, invert=False)
        tracks = tp.link_df(features, 15, memory=10)
        save_tracked_data(filename[:-4] + '_out.dat', frames, tracks, times, laser_powers, traps)
        with open(filename[:-4] + '_out.dat', 'r') as calculated_file:
            with open(filename[:-4] + '_expected.dat', 'r') as expected_file:
                for calculated, expected in zip(calculated_file, expected_file):
                    self.assertEqual(calculated, expected)
                    
if __name__ == "__main__":
    unittest.main()
