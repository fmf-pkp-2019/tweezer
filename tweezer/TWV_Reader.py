"""
Defines the TWV_Reader for pims.
The reader reads .twv files from Optical Tweezers.
Requirements:
 - ctypes
 - pims
 - numpy
Optional requirements:
 - matplotlib
"""

from ctypes import c_uint16, c_uint32, c_float, c_double
from ctypes import Structure
import struct
import unittest
from pims import FramesSequence, Frame
import numpy as np

VERBOSE_LEVEL = 0

class TArframe_ROI(Structure):
    _pack_ = 1
    _fields_ = [("Left", c_uint16),
                ("Top", c_uint16),
                ("Width", c_uint16),
                ("Height", c_uint16)
                ]

class TArFrameData(Structure):
    _pack_ = 1
    _fields_ = [("HeaderSize", c_uint16),
                ("FrameDataIncl", c_uint16),
                ("ROI", TArframe_ROI),
                ("BytesPerPixel", c_uint16),
                ("FrameRate", c_double),
                ("Exposure", c_double),
                ("Gain", c_double),
                ("RecTrapDataNo", c_uint16),
                ]

class TArCalibrationData(Structure):
    _pack_ = 1
    _fields_ = [("ImageToSampleScale", c_float)
                ]

class TArvideo_header(Structure):
    _pack_ = 1
    _fields_ = [("VideoID", c_uint16),
                ("VideoVersion", c_uint16),
                ("VideoLicence", c_uint16),
                ("VideoHeaderSize", c_uint32),
                ("RecordedFrames", c_uint32),
                ("FrameData", TArFrameData),
                ("CalibrationData", TArCalibrationData)
                ]

class TArTrapData(Structure):
    _pack_ = 1
    _fields_ = [("PositionX", c_float),
                ("PositionY", c_float),
                ("Intensity", c_float)
                ]

class TArframe_header(Structure):
    _pack_ = 1
    _fields_ = [("FrameNumber", c_uint32),
                ("FrameTime", c_float),
                ("LaserPower", c_float),
                ("CalibrationData1", TArTrapData),
                ("CalibrationData2", TArTrapData),
                ("CalibrationData3", TArTrapData),
                ("CalibrationData4", TArTrapData)
                ]

def display_attr(obj):
    """
    Prints obj. Used for testing.
    """
    for field_name, field_type in obj._fields_:
        print(field_name, getattr(obj, field_name))

class TWV_Reader(FramesSequence):
    """
    Reader for .twv files for pims package.
    """

    def __init__(self, filename):
        """
        Inputs:
         - filename: Filename string, ending with .twv.

        Reads video header and prepares video header data.
        """
        self.filename = filename
        self.f = open(filename, "rb")

        self.video_header = TArvideo_header()
        self.f.readinto(self.video_header)
        self.f.seek(getattr(self.video_header, "VideoHeaderSize"))
        if self.video_header.FrameData.BytesPerPixel != 1:
            from warnings import warn
            warn("video_header.FrameData.BytesPerPixel is not 1")

        self._len = self.video_header.RecordedFrames
        self._dtype = np.uint8
        self._frame_shape = (self.video_header.FrameData.ROI.Width,
                             self.video_header.FrameData.ROI.Height)
        self.frame_header_size = self.video_header.FrameData.HeaderSize
        self.frame_size_bytes = (self._frame_shape[0] * self._frame_shape[1] +
                                 self.frame_header_size)

    def get_all_metadata(self):
        """
        Returns video metadata (not individual frame metadata) as a dict.
        """
        if 0:
            display_attr(self.video_header)
            print('---')
            display_attr(self.video_header.FrameData)
            print('---')
            display_attr(self.video_header.FrameData.ROI)
            print('---')
            display_attr(self.video_header.CalibrationData)

        out_dict = dict()
        out_dict['ROI'] = dict()
        out_dict['ROI']['Left'] = self.video_header.FrameData.ROI.Left
        out_dict['ROI']['Top'] = self.video_header.FrameData.ROI.Top
        out_dict['ROI']['Width'] = self.video_header.FrameData.ROI.Width
        out_dict['ROI']['Height'] = self.video_header.FrameData.ROI.Height
        out_dict['ImageToSampleScale'] = self.video_header.CalibrationData.ImageToSampleScale
        out_dict['FrameRate'] = self.video_header.FrameData.FrameRate
        out_dict['Exposure'] = self.video_header.FrameData.Exposure
        out_dict['Gain'] = self.video_header.FrameData.Gain
        return out_dict

    def set_end_frame(self, end_frame):
        """
        Slicing objects in pims should work, but doesn't.
        This function defines 'end_frame' as the last frame of the video.
        """
        if end_frame <= self.video_header.RecordedFrames:
            self._len = end_frame
        else:
            raise ValueError("Unable to set {:} as the end frame.".format(end_frame) +
                             "There are only {:} frames in the file.".format(
                                 self.video_header.RecordedFrames))

    def reset_end_frame(self):
        """ Resets frame count to default value (the same as in the opened file). """
        self._len = self.video_header.RecordedFrames

    def get_frame(self, frame_no):
        """
        Returns a frame (image) as a np.array.
        """
        self.f.seek(self.video_header.VideoHeaderSize) # absolute seek
        self.f.seek(frame_no * self.frame_size_bytes + self.frame_header_size, 1) # relative seek
        image = []
        unpack_format = '{:}B'.format(self._frame_shape[0])
        for i in range(self._frame_shape[1]):
            image.append(struct.unpack(
                unpack_format,
                self.f.read(self._frame_shape[0])))
        image = np.array(image, dtype=self._dtype)

        return Frame(image, frame_no=frame_no)

    def get_frame_time(self, frame_no):
        """
        Returns the timestamp of a frame.
        """
        self.f.seek(self.video_header.VideoHeaderSize) # absolute seek
        self.f.seek(frame_no * self.frame_size_bytes, 1) # relative seek
        frame_header = TArframe_header()
        self.f.readinto(frame_header)
        return frame_header.FrameTime

    def __len__(self):
        return self._len

    def __del__(self):
        """
        Closes the file on object destruction.
        Also seems to close the file when program crashes.
        """
        self.f.close()

    @property
    def frame_shape(self):
        """
        Mandatory for pims.
        """
        return self._frame_shape

    @property
    def pixel_type(self):
        """
        Mandatory for pims.
        """        
        return self._dtype

    @classmethod
    def class_exts(cls):
        """
        Mandatory for pims.
        """
        return {'twv'} | super(TWV_Reader, cls).class_exts()

    def check_for_time_jumps(self, treshold=10**-2, show=False):
        """
        Checks if there are any unusual time jumps in the file.
        Returns True if data is OK max_time > frame_time * (1 + treshold) or
            min_time < frame_time * (1 - treshold)
        Kwargs:
         - treshold=10**-2: If relative error is below this treshold,
             video is considered OK.
         - show=False: If true, the time vs. frame number and time step are plotted
             (matplotlib required).

        """
        times = []
        self.f.seek(self.video_header.VideoHeaderSize) # absolute seek
        frame_rate = self.video_header.FrameData.FrameRate
        frame_time = 1/frame_rate
        for _ in range(self._len):
            frame_header = TArframe_header()
            self.f.readinto(frame_header)

            times.append(frame_header.FrameTime)
            self.f.seek(self._frame_shape[0] * self._frame_shape[1], 1) # relative seek

        times = np.array(times)
        time_deltas = times[1:]-times[:-1]
        min_time = min(time_deltas)
        max_time = max(time_deltas)
        if VERBOSE_LEVEL >= 2:
            print("Min frame time {:}, max frame time {:}, frame_time {:}".format(
                  min_time, max_time, frame_time))
        ret = True
        if (max_time > frame_time * (1 + treshold) or
            min_time < frame_time * (1 - treshold)):
            if VERBOSE_LEVEL >= 1:
                print("There are time jumps: min_time {:}, max_time {:},"
                      "relative difference {:}, treshold {:}".format(
                        min_time, max_time, (max_time-min_time)/min_time, treshold))
            ret = False
        if show:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(range(len(times)), times)
            ax[0].set_title("Time vs frame number")
            ax[0].set_ylabel("Time from start [s]")
            
            ax[1].plot(range(len(times)-1), time_deltas, label="Actual time between frames")
            ax[1].plot([0, len(times)-1], [frame_time, frame_time],
                       label="Reported time between frames")
            ax[1].legend()
            ax[1].semilogy()
            ax[0].set_title("avg FPS = {:}, reported FPS = {:}".format(
                len(times)/(times[-1] - times[0]), frame_rate))
            ax[1].set_ylabel("Time between frames [s]\nlog scale")
            ax[1].set_xlabel("Frame number")
            plt.show()
            plt.cla()
            plt.clf()
            
        return ret

    def get_all_tweezer_positions(self, which=[0,1,2,3], fname=None):
        """
        TODO - cleanup
        Returns all tweezers positions
        - [times, laserPower, (trapX, trapY, intensity) for each trap]
        Optional input: which:, specifies which tweezers positions to return.
        If fname is set, it is written to it (tab separated values).
        Otherwise it is returned as an array.
        """
        if VERBOSE_LEVEL > 1:
            print("get_all_tweezer_positions")
        laser_powers = [] # TODO
        times = []
        traps = [[] for i in range(4)]
        self.f.seek(self.video_header.VideoHeaderSize) # absolute seek
        for _ in range(len(self)):
            frame_header = TArframe_header()
            self.f.readinto(frame_header)
            times.append(frame_header.FrameTime)
            laser_powers.append(frame_header.LaserPower)
            for j in which:
                tmp = frame_header.__getattribute__(
                    "CalibrationData{:}".format(j+1))
                traps[j].append([
                    tmp.PositionX,
                    tmp.PositionY,
                    tmp.Intensity])
            self.f.seek(self._frame_shape[0] * self._frame_shape[1], 1) # relative seek

        times = np.array(times)
        if fname is None:
            return times, laser_powers, traps
            # times, laser_powers, traps[which_trap][time][x, y, power]
        else:
            pass
            # TODO
            # this will not be used. This will be used together
            # with particle tracker.

class Test(unittest.TestCase):
    """
    Additional tests of this module happen in particle tracker module.
    Most notably and end-to-end test.
    """

    def setUp(self):
        #self.twv_reader = TWV_Reader("passiveInTrapP1.twv")
        self.filename = "../examples/data/test_example.twv"

    def test_set_end_frame_number(self):
        twv_reader = TWV_Reader(self.filename)
        twv_reader.set_end_frame(10)
        self.assertEqual(len(twv_reader), 10)

    @unittest.expectedFailure
    def test_check_end_frame_too_high(self):
        twv_reader = TWV_Reader(self.filename)
        twv_reader.set_end_frame(100000)
        twv_reader.f.close()

    def test_check_metadata(self):
        twv_reader = TWV_Reader(self.filename)
        calculated = twv_reader.get_all_metadata()
        expected = {'ROI': {'Height': 194, 'Width': 216, 'Left': 976, 'Top': 862},
                    'FrameRate': 20.0, 'Gain': 1.0, 'Exposure': 1.9845360000000003,
                    'ImageToSampleScale': 10.666666984558105}
        self.assertEqual(expected, calculated)

def example(filename):
    """
    Example usega of TWV_Reader.
    """
    frames = TWV_Reader(filename)

    metadata = frames.get_all_metadata()

    frames.check_for_time_jumps(show=False)

    times, laser_powers, traps = frames.get_all_tweezer_positions()

    #import cv2
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #for i in range(a.video_header.RecordedFrames):
    #    cv2.imshow('image', a[i])
    #    cv2.waitKey(1)

if __name__ == '__main__' and 1:
    example("../examples/data/test_example.twv")

if __name__ == '__main__':
    unittest.main()

