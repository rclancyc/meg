import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy import linalg as la
from numpy import sin, cos, tan

def get_cartesian_coords(_r, _t, _p):
    # Helper function to convert from spherical into cartesian coords.
    cart_coords = _r * np.array((np.sin(_p) * np.cos(_t), np.sin(_p) * np.sin(_t), np.cos(_p)))
    return cart_coords

def get_spherical_coords(x, y, z):
    """
    :param x value
    :param y value
    :param z value Cartestian coordinates
    :return: rho, theta, and phi angles in spherical. theta is polar angle and phi is azimuthal.
    """
    rho = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / rho)
    r = rho*np.sin(phi)

    if np.abs(r) < 1.0e-16:
        theta = 0
    else:
        if np.abs(x/r) <= 1:
            theta = np.arccos(x / r)
        else:
            theta = (0 if x/r > 1 else np.pi)
        theta = 2*np.pi - theta if y < 0 else theta
    return np.array([rho, theta, phi])


def get_rotation_matrix(angx, angy, angz, deg=True):
    if deg:
        angx = np.pi * angx / 180
        angy = np.pi * angy / 180
        angz = np.pi * angz / 180
    Rx = np.array([[1,0,0],[0,cos(angx),-sin(angx)],[0,sin(angx),cos(angx)]])
    Ry = np.array([[cos(angy),0,sin(angy)],[0,1,0],[-sin(angy),0,cos(angy)]])
    Rz = np.array([[cos(angz),-sin(angz),0],[-sin(angz),cos(angz),0],[0,0,1]])

    return np.matmul(Rx,np.matmul(Ry,Rz))






def plot3dscatter(sensors):
    fig = plt.figure(17)
    ax = plt.axes(projection='3d')
    radius = 0.131
    mesh_center = [0,0,0]
    # plots fields at sensors
    ax.scatter(sensors[:, 0], sensors[:, 1], sensors[:, 2], color='r')

    ax.set_xlim([-1 * radius, 1 * radius])
    ax.set_ylim([-1 * radius, 1 * radius])
    ax.set_zlim([0, 1 * radius])
    plt.show()
