"""A simple matlotlib-based video viewer. Video can be a list-like object or an
iterator that yields 2D array."""
from __future__ import absolute_import, print_function, division 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class VideoViewer(object):
    """
    A matplotlib-based video viewer.
    
    Parameters
    ----------
    video : list-like, iterator
        A list of 2D arrays or a generator of 2D arrays. If an iterator is
        provided, you must set nframes as well. 
    nframes: int, optional
        How many frames to show.
    """

    def __init__(self, video, nframes = None, title = ""):

        if nframes is None:
            try:
                nframes = len(video)
            except TypeError:
                raise Exception("You must specify nframes!")
        self.index = 0
        self.video = video
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        plt.subplots_adjust(bottom=0.25)
        
        frame = iter(video).__next__() #take first frame
        self.img = self.ax.imshow(frame) 

        self.axframe= plt.axes([0.25, 0.1, 0.65, 0.03])
        self.sframe = Slider(self.axframe, 'Frame', 0, nframes - 1, valinit=0, valstep=1, valfmt='%i')
        
        def update(val):
            i = int(self.sframe.val)
            try:
                frame = self.video[i] #assume list-like object
                self.index = i
            except TypeError:
                #assume generator
                frame = None
                if i > self.index:
                    for frame in self.video:
                        self.index += 1
                        if self.index >= i:
                            break
            if frame is not None:
                self.img.set_data(frame)
                self.fig.canvas.draw_idle()
    
        self.sframe.on_changed(update)
        
    def show(self):
        """Shows video."""
        self.fig.show()
        
if __name__ == "__main__":
    video = (np.random.randn(256,256) for i in range(256))
    vg = VideoViewer(video, 256, title = "iterator example") #must set nframes, because video has no __len__
    vg.show()
    
    video = [np.random.randn(256,256) for i in range(256)] 
    vl = VideoViewer(video, title = "list example") 
    vl.show()   
    