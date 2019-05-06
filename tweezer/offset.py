import plotting 
import numpy as np

def four_corner_offsets(upper_left, upper_right, lower_left, lower_right):
    """Reads particle and trap positions of four measurements and returns particle-trap offsets.
    To ensure good results, the positions should roughly form a rectangle. 

    Parameters
    ----------
    upper_left : str
        name of 1st file to open
    upper_right : str
        name of 2nd file to open
    lower_left : str
        name of 3rd file to open
    lower_right : str
        name of 4th file to open

    Returns
    -------
    points : ndarray_like
        n-th row stores x-,y-coordinates of n-th trap and x-,y-offset of n-th particle
    """

    points = np.zeros((4,4))

    _,traps,particles = plotting.read_file(upper_left,1)
    points[0,:] = (np.mean(traps[:,0]), np.mean(traps[:,1]) ,np.mean(particles[:,0]) - np.mean(traps[:,0]),np.mean(particles[:,1]) - np.mean(traps[:,1]))
    _,traps,particles = plotting.read_file(upper_right,1)
    points[0,:] = (np.mean(traps[:,0]), np.mean(traps[:,1]) ,np.mean(particles[:,0]) - np.mean(traps[:,0]),np.mean(particles[:,1]) - np.mean(traps[:,1]))
    _,traps,particles = plotting.read_file(lower_left,1)
    points[0,:] = (np.mean(traps[:,0]), np.mean(traps[:,1]) ,np.mean(particles[:,0]) - np.mean(traps[:,0]),np.mean(particles[:,1]) - np.mean(traps[:,1]))
    _,traps,particles = plotting.read_file(lower_right,1)
    points[0,:] = (np.mean(traps[:,0]), np.mean(traps[:,1]) ,np.mean(particles[:,0]) - np.mean(traps[:,0]),np.mean(particles[:,1]) - np.mean(traps[:,1]))

    return points

def four_corner_calibration(trap_pos_x, trap_pos_y, points):
    """Interpolates an approximate particle offset at chosen position, given offsets at four calibration positions.
    Separate from four_corner_offsets to avoid redundant calculations when using multiple positions. 

    Parameters
    ----------
    trap_pos_x : float
        x-coordinate of point for which offset is calculated
    trap_pos_y : str
        y-coordinate of point for which offset is calculated
    points : ndarray_like
        x-,y-coordinates of traps and x-,y-offsets at all four positions

    Returns
    -------
    offset_x : float
        approximate particle-trap offset in x-direction at chosen position
    offset_y : float
        approximate particle-trap offset in y-direction at chosen position

    Examples
    --------
    TODO
    """

    p1 = points[0,:]
    p2 = points[1,:]
    p3 = points[2,:]
    p4 = points[3,:]

    CX = (trap_pos_y - 0.5*(p3[1]+p4[1]))/(0.5*(p1[1]+p2[1])-0.5*(p3[1]+p4[1]))
    CY = (trap_pos_x - 0.5*(p1[0]+p3[0]))/(0.5*(p2[0]+p4[0])-0.5*(p1[0]+p3[0]))

    offset_x = CX*(p1[2]+(p2[2]-p1[2])*(trap_pos_x - p1[0])/(p2[0] - p1[0])) + (1.0-CX)*(p3[2]+(p4[2]-p3[2])*(trap_pos_x-p3[0])/(p4[0]-p3[0]))
    offset_y = (1.0-CY)*(p3[3]+(p1[3]-p3[3])*(trap_pos_y - p3[1])/(p1[1] - p3[1])) + (CY)*(p4[3]+(p2[3]-p4[3])*(trap_pos_y-p4[1])/(p2[1]-p4[1]))

    return offset_x, offset_y
