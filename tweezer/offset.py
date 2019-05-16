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

    _,trap_1,traj_1 = plotting.read_file(upper_left,1)
    _,trap_2,traj_2 = plotting.read_file(upper_right,1)
    _,trap_3,traj_3 = plotting.read_file(lower_left,1)
    _,trap_4,traj_4 = plotting.read_file(lower_right,1)

    points = four_corner_offsets_calculate(trap_1,trap_2,trap_3,trap_4,traj_1,traj_2,traj_3,traj_4)

    return points

def four_corner_offsets_calculate(trap_1, trap_2, trap_3, trap_4, pos_1, pos_2, pos_3, pos_4):
    """Calculates particle-trap offsets. To ensure good results, the positions should roughly form a rectangle. 
    Parameters
    ----------
    trap_1 : ndarray_like
        positions of traps for 1st measurement
    trap_2 : ndarray_like
        positions of traps for 2nd measurement
    trap_3 : ndarray_like
        positions of traps for 3rd measurement
    trap_4 : ndarray_like
        positions of traps for 4th measurement
    pos_1 : ndarray-like
        trajectory of particle for 1st measurement
    pos_2 : ndarray-like
        trajectory of particle for 2nd measurement
    pos_3 : ndarray-like
        trajectory of particle for 3rd measurement
    pos_4 : ndarray-like
        trajectory of particle for 4th measurement
    Returns
    -------
    points : array of float
        n-th row stores x-,y-coordinates of n-th trap and x-,y-offset of n-th particle
    """

    points = np.zeros((4,4))

    points[0,:] = (np.mean(trap_1[:,0]), np.mean(trap_1[:,1]) ,np.mean(pos_1[:,0]) - np.mean(trap_1[:,0]),np.mean(pos_1[:,1]) - np.mean(trap_1[:,1]))
    points[1,:] = (np.mean(trap_2[:,0]), np.mean(trap_2[:,1]) ,np.mean(pos_2[:,0]) - np.mean(trap_2[:,0]),np.mean(pos_2[:,1]) - np.mean(trap_2[:,1]))
    points[2,:] = (np.mean(trap_3[:,0]), np.mean(trap_3[:,1]) ,np.mean(pos_3[:,0]) - np.mean(trap_3[:,0]),np.mean(pos_3[:,1]) - np.mean(trap_3[:,1]))
    points[3,:] = (np.mean(trap_4[:,0]), np.mean(trap_4[:,1]) ,np.mean(pos_4[:,0]) - np.mean(trap_4[:,0]),np.mean(pos_4[:,1]) - np.mean(trap_4[:,1]))

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
