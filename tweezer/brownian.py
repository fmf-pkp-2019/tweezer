"""
Brownian motion simulator for two-dimensional Brownian motion of spherical particles.

You can use this script to simulate brownian motion. You define number of particles
to simulate and the region of interest (simulation box). Particles are randomly
placed in the box, simulation is done using periodic boundary conditions - when 
the particle reaches the edge of the frame it is mirrored on the other side. 
Then optical microscope image capture is simumated by writing a point spread function to the 
image at the calculated particle positions.

The frame grabber is done using iterators to reduce memory requirements. You can
analyze video frame by frame with minimal memory requirement, or read the whole
video first to memory.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import numba as nb
import math

PARALLEL = False #set to True if you want to compile for parallel 
SINGLE_PRECISION = False #set this to True if you want to calculate in single precision

TARGET = "parallel" if PARALLEL == True else "cpu"

if SINGLE_PRECISION:
    F = nb.float32
    I = nb.int32
    
    FDTYPE = np.float32
    IDTYPE = np.int32
    
else:
    F = nb.float64
    I = nb.int64
    
    FDTYPE = np.float64
    IDTYPE = np.int64
    
U8 = nb.uint8 #imaging is done in 8bit mode... 
 
@nb.vectorize([F(F,F,F)], target = TARGET)
def mirror(x,x0,x1):
    """transforms coordinate x by flooring in the interval of [x0,x1]
    It performs x0 + (x-x0)%(x1-x0)"""
    #return x0 + (x-x0)%(x1-x0)
    
    # Most of the time there will be no need to do anything, 
    # so the implementation below is faster than floor implementation above
    
    if x0 == x1:
        return x
    while True: 
        if x < x0:
            x = x1 + x - x0
        elif x >= x1:
            x = x0 + x - x1
        else:
            return x
                          
@nb.vectorize([F(F,F,F)], target = TARGET)        
def make_step(x,scale, velocity):
    """Performs random particle step from a given initial position x."""
    return x + np.random.randn()*scale + velocity   

def brownian_walk(x0, n = 1024, shape = (256,256), delta = 1, dt = 1, velocity = 0.):
    """Returns an brownian walk iterator.
     
    Given the initial coordinate x0, it callculates next n coordinates."""             
    particles, xy = x0.shape
    scale=delta*np.sqrt(dt)
    x = x0
    x = np.asarray(x, FDTYPE)
    velocity = np.asarray(velocity, FDTYPE)/dt
    scale = np.asarray(scale, FDTYPE)
    
    x1, x2 = 0, np.asarray(shape,FDTYPE)

    for i in range(n):
        yield x
        x = make_step(x,scale, velocity)
        x = mirror(x,x1,x2)
        
def brownian_particles(n = 500, shape = (256,256),particles = 10,delta = 1, dt = 1,velocity = 0.):
    """Creates coordinates of multiple brownian particles.
    
    Parameters
    ----------
    n : int
        Number of steps to calculate
    shape : (int,int)
        Shape of the box
    particles : int
        Number of particles in the box
    delta : float
        Step variance in pixel units (when dt = 1)
    dt : float
        Time resolution
    velocity : float
        Velocity in pixel units (when dt = 1) 
    """
    x0 = np.asarray(np.random.rand(particles,2)*np.array(shape),FDTYPE)
    v0 = np.zeros_like(x0)
    v0[:,0] = velocity
    for data in brownian_walk(x0,n,shape,delta,dt,v0):
        yield data
             
GAUSSN = 1/np.sqrt(2*np.pi)

@nb.jit([U8(I,F,I,F,F,U8)],nopython = True)
def psf_gauss(x,x0,y,y0,sigma,intensity):
    """Gaussian point-spread function. This is used to calculate pixel value
    for a given pixel coordinate x,y and particle position x0,y0."""
    return intensity*math.exp(-0.5*((x-x0)**2+(y-y0)**2)/(sigma**2))
                 
@nb.jit(["uint8[:,:],float64[:,:],uint8"], nopython = True)                
def draw_points(im, points, intensity):
    """Draws pixels to image from a given points array"""
    data = points
    particles = len(data)
    for j in range(particles):
        im[int(data[j,0]),int(data[j,1])] = im[int(data[j,0]),int(data[j,1])] + intensity 
    return im   
        
@nb.jit([U8[:,:](U8[:,:],F[:,:],U8,F)], nopython = True, parallel = PARALLEL)                
def draw_psf(im, points, intensity, sigma):
    """Draws psf to image from a given points array"""
    height, width = im.shape
    particles = len(points)
    size = int(round(3*sigma))
    for k in nb.prange(particles):
        h0,w0  = points[k,0], points[k,1]
        h,w = int(h0), int(w0) 
        for i0 in range(h-size,h+size+1):
            for j0 in range(w -size, w+size+1):
                p = psf_gauss(i0,h0,j0,w0,sigma,intensity)
                #j = j0 % width
                #i = i0 % height
                
                # slightly faster implementation of flooring
                if j0 >= width:
                    j= j0 - width
                elif j0 < 0:
                    j = width + j0
                else:
                    j = j0
                if i0>= height:
                    i = i0 - height
                elif i0 < 0:
                    i = height + i0
                else:
                    i = i0
                im[i,j] = im[i,j] + p
    return im  
 
def particles_video(particles, shape = (512,512),
                 background = 0, intensity = 10, sigma = None):
    """Creates brownian particles video"""
        
    background = np.zeros(shape = shape,dtype = "uint8") + background
    height, width = shape

    def get_frame(data):
        im = background.copy()
        if sigma is None:
            im = draw_points(im, data, intensity)
        else:
            im = draw_psf(im, data, intensity, sigma)
        return im
        
    for i,data in enumerate(particles):
        yield get_frame(data)

def test_plot(n = 5000, particles = 2):
    """Brownian particles usage example. Track 2 particles"""
    import matplotlib.pyplot as plt 
    x = np.array([x for x in brownian_particles(n = n, particles = particles)])
    plt.figure()
    
    for i in range(particles): 
        
        # Plot the 2D trajectory.
        plt.plot(x[:,i,0],x[:,i,1])
        
        # Mark the start and end points.
        plt.plot(x[0,i,0],x[0,i,1], 'go')
        plt.text(x[0,i,0],x[0,i,1], str(i))
        plt.plot(x[-1,i,0], x[-1,i,1], 'ro')
        plt.text(x[-1,i,0], x[-1,i,1], str(i))
        
    
    # More plot decorations.
    plt.title('2D Brownian Motion with mirror boundary conditions.')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def frame_grabber(nframes, shape = (256,256), intensity = 30, sigma = 2, **kw):
    kw["n"] = nframes
    kw["shape"] = shape
    p = brownian_particles(**kw) 
    return particles_video(p, shape = shape, sigma = sigma, intensity = intensity) 

if __name__ == "__main__":
    video = frame_grabber(1024, dt = 0.1) #this is an iterator
    import viewer
    video = list(video) #lets read it into memort by creating a list
    v1 = viewer.VideoViewer(video) 
    v1.show()
    test_plot()