import math
import random
import statistics
import numpy as np
import scipy.constants as constants

def SAT1(fileName, numPoints, dt, kTrap, trapFrequency, trapAmplitude, beadRadius, eta):
    #   Originally implemented in MATLAB by Natan Osterman, advised by Andrej Vilfan, 23.3.2012.
    #
    #   Simulates the Brownian motion of a colloidal bead trapped in an optical trap oscillating in the x direction.
    #   for a given numPoints value, writes a file of numPoints rows and 4 columns; returns two variables (trap stiffnesses kx, ky)
    #
    #   Syntax:
    #   function [poz]=  SAT1(fileName, numPoints, dt, kTrap, trapFrequency, trapAmplitude, beadRadius, eta)
    #
    #   dt: time interval between two consecutive positions [s]
    #   kTrap: trap stiffness [N/m]
    #   trapFrequency: trap oscillation frequency [Hz]
    #   trapAmplitude: amplitude of oscillation [m]
    #   beadRadius: radius of trapped particle [m]
    #   eta: viscosity of medium [Pa s]
    #
    #   Warning: position values in output file are in micrometers!

    print("Calculating ...")
    fout = open(fileName, "w")

    kBT = constants.Boltzmann*300   #   assumption: T=300K
    a = beadRadius
    x = np.array([0.,0.])
    xTrap=np.array([0.,0.])
    noise = np.array([0.,0.])
    dx = np.array([0.,0.])
    dtInternal=0.0002   #   internal time step used for simulation [s]

    poz = np.zeros([numPoints,2])   #   initial position of bead [x y] in meters
    trapPoz = np.zeros([numPoints,2])   #   initial position of trap
    time = np.zeros([numPoints])

    i=0
    t=0
    lastSampleInterval=dt+1e-10

    while i < numPoints:
        if lastSampleInterval > dt:
            lastSampleInterval -= dt
            i += 1
            fout.write("%d %3.3f %3.3f %3.3f\r\n" % (i,t,x[0]*1e6,x[1]*1e6))
            poz[i-1,:] = x
            trapPoz[i-1,:] = xTrap
            time[i-1] = t
        t += dtInternal
        lastSampleInterval += dtInternal
        noise = [(2*random.random()-1)*math.sqrt(3),(2*random.random()-1)*math.sqrt(3)]
        xTrap = [trapAmplitude*math.sin(2*math.pi*trapFrequency*t),0]
        dx = ((-kTrap*dtInternal)/(6*math.pi*eta*a))*(np.array(x)-np.array(xTrap))+(math.sqrt(2*kBT/(6*math.pi*eta*a))*math.sqrt(dtInternal))*np.array(noise)
        x += dx
    
    fout.close()

    kCalculated = np.zeros([numPoints,2])
    for i in range(0,numPoints):
        kCalculated[i,0] = poz[i,0]*poz[i,0]
        kCalculated[i,1] = poz[i,1]*poz[i,1]

    kxCalculated=kBT/statistics.mean(kCalculated[:,0])
    kyCalculated=kBT/statistics.mean(kCalculated[:,1])

    return kxCalculated*1e6, kyCalculated*1e6

def SAT2(fileName, numPoints, dt, kxTrap, kyTrap, xTrapFreq, yTrapFreq, xTrapAmp, yTrapAmp, beadRadius, eta, temp=293, motionType=1):

    #   Simulates the Brownian motion of a colloidal bead trapped in an optical trap oscillating in x and y directions.
    #
    #   Syntax: same as SAT1, except for the following
    #
    #   temp: Temperature in Kelvin
    #   motionType: governs trap motions and can take values 1 or 2.
    #       1: sinusoidal motion in x,y (default)
    #       2: linear motion in x,y; in this case, trap frequency parameters are ignored
    #           and the amplitudes become velocities in [m/s]

    dtInternal=0.0001   #   internal time step used for simulation [s]

    if (dt <= dtInternal):
        raise ValueError("dt must be longer than time step of simulation")

    print("\nCalculating ...")
    fout = open(fileName, "w")

    kBT = constants.Boltzmann*temp   #   assumption: T=300K
    a = beadRadius
    x = np.array([0.,0.])
    xTrap=np.array([0.,0.])
    kTrap = np.array([kxTrap,kyTrap])
    noise = np.array([0.,0.])
    dx = np.array([0.,0.])
    
    poz = np.zeros([numPoints,2])   #   initial position of bead [x y] in meters
    trapPoz = np.zeros([numPoints,2])   #   initial position of trap
    time = np.zeros([numPoints])

    i=0
    t=0
    lastSampleInterval=dt+1e-10

    while i < numPoints:
        if lastSampleInterval > dt:
            lastSampleInterval -= dt
            i += 1
            fout.write("%d %3.3f %3.3f %3.3f %3.3f %3.3f\r\n" % (i,t,x[0]*1e6,x[1]*1e6,xTrap[0]*1e6,xTrap[1]*1e6))
            poz[i-1,:] = x
            trapPoz[i-1,:] = xTrap
            time[i-1] = t
        t += dtInternal
        lastSampleInterval += dtInternal
        noise = [(2*random.random()-1)*math.sqrt(3),(2*random.random()-1)*math.sqrt(3)]

        if (motionType == 1):
            xTrap = [xTrapAmp*math.sin(2*math.pi*xTrapFreq*t),yTrapAmp*math.cos(2*math.pi*yTrapFreq*t)]
        elif (motionType == 2):
            xTrap = [xTrapAmp*t,yTrapAmp*t]

        dx = (dtInternal/(6*math.pi*eta*a))*np.array(kTrap)*(np.array(xTrap)-np.array(x))+(math.sqrt(2*kBT/(6*math.pi*eta*a))*math.sqrt(dtInternal))*np.array(noise)
        x += dx
    
    fout.close()

    kEstimate = np.zeros([numPoints,2])
    for i in range(0,numPoints):
        kEstimate[i,0] = poz[i,0]*poz[i,0]
        kEstimate[i,1] = poz[i,1]*poz[i,1]

    kxEstimate=kBT/statistics.mean(kEstimate[:,0])*1e6
    kyEstimate=kBT/statistics.mean(kEstimate[:,1])*1e6

    return kxEstimate, kyEstimate