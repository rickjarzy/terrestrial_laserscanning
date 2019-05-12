import scipy
import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def create_tls_data(m_x, m_y, m_z, radius, verbose=False):
    # zylinder as stativ aproximation
    radius_zylinder = radius / 3
    height = radius * 3

    x_coords_zyl = []
    y_coords_zyl = []
    z_coords_zyl = []

    x_coords = []
    y_coords = []
    z_coords = []

    for phi in range(1, 360, 12):
        for theta in range(1, 360, 12):
            x = m_x + radius * numpy.sin(theta * (numpy.pi / 180)) * numpy.cos(phi * (numpy.pi / 180))
            y = m_y + radius * numpy.sin(theta * (numpy.pi / 180)) * numpy.sin(phi * (numpy.pi / 180))
            z = m_z + radius * numpy.cos(theta * (numpy.pi / 180))
            check = numpy.sqrt((m_x - x) ** 2 + (m_y - y) ** 2 + (m_z - z) ** 2)

            if verbose:
                print("phi %s - theta %s " % (phi, theta))
                print("X: %s, - Y: %s - Z: %s" % (x, y, z))
                print("check radius: ", check)

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

    for phi in range(1, 360, 12):
        for h in range(m_z - height, m_z - radius, 1):
            x = m_x + radius_zylinder * numpy.cos(phi * (numpy.pi / 180))
            y = m_y + radius_zylinder * numpy.sin(phi * (numpy.pi / 180))
            z = h

            x_coords_zyl.append(x)
            y_coords_zyl.append(y)
            z_coords_zyl.append(z)


    x_arr = numpy.asarray(x_coords) + numpy.random.rand(len(x_coords))
    y_arr = numpy.asarray(y_coords) + numpy.random.rand(len(x_coords))
    z_arr = numpy.asarray(z_coords) + numpy.random.rand(len(x_coords))

    x_arr_zyl = numpy.asarray(x_coords_zyl) + numpy.random.rand(len(x_coords_zyl))
    y_arr_zyl = numpy.asarray(y_coords_zyl) + numpy.random.rand(len(x_coords_zyl))
    z_arr_zyl = numpy.asarray(z_coords_zyl) + numpy.random.rand(len(x_coords_zyl))

    if verbose:
        print(" len of x ", len(x_coords))
        # 3d Scattrer plot - FULL resolution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter3D(x_arr, y_arr, z_arr, c=z_arr, cmap='jet', linewidths=0.5, label="Test Sphere Points")
        ax.scatter3D(x_arr_zyl, y_arr_zyl, z_arr_zyl, c=z_arr_zyl, cmap='viridis', linewidths=0.5,
                     label="Test Zylinder Points")
        ax.set_title("3D Scatter Plot of the Test Sphere")
        ax.set_xlabel("X Coorinates")
        ax.set_ylabel("Y Coorinates")
        ax.set_zlabel("Z Coorinates")
        ax.legend()

        plt.show()
    x_data = numpy.hstack((x_arr, x_arr_zyl))
    y_data = numpy.hstack((y_arr, y_arr_zyl))
    z_data = numpy.hstack((z_arr, z_arr_zyl))

    return x_data, y_data, z_data,

if __name__ == "__main__":

    # sphere definition
    sphere_radius = 12
    sphere_x = 12
    sphere_y = 16
    sphere_z = 99

    tls_x, tls_y, tls_z = create_tls_data(sphere_x, sphere_y, sphere_z, sphere_radius, True)






    print("Programm ENDE")