import scipy
import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def create_tls_data(m_x, m_y, m_z, radius, verbose=False, stativ=False):
    # zylinder as stativ aproximation
    radius_zylinder = radius / 3
    height = radius * 3



    x_coords = []
    y_coords = []
    z_coords = []

    #sphere
    for phi in range(1, 180, 12):
        for theta in range(1, 180, 12):
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

    x_arr = numpy.asarray(x_coords) + numpy.random.rand(len(x_coords))*1.5
    y_arr = numpy.asarray(y_coords) + numpy.random.rand(len(x_coords))*1.5
    z_arr = numpy.asarray(z_coords) + numpy.random.rand(len(x_coords))*1.5

    #stativ
    if stativ:
        x_coords_zyl = []
        y_coords_zyl = []
        z_coords_zyl = []
        for phi in range(1, 180, 12):
            for h in range(m_z - height, m_z - radius, 1):
                x = m_x + radius_zylinder * numpy.cos(phi * (numpy.pi / 180))
                y = m_y + radius_zylinder * numpy.sin(phi * (numpy.pi / 180))
                z = h

                x_coords_zyl.append(x)
                y_coords_zyl.append(y)
                z_coords_zyl.append(z)

                x_arr_zyl = numpy.asarray(x_coords_zyl) + numpy.random.rand(len(x_coords_zyl)) * 1.5
                y_arr_zyl = numpy.asarray(y_coords_zyl) + numpy.random.rand(len(x_coords_zyl)) * 1.5
                z_arr_zyl = numpy.asarray(z_coords_zyl) + numpy.random.rand(len(x_coords_zyl)) * 1.5



    if verbose:
        print(" len of x ", len(x_coords))
        # 3d Scattrer plot - FULL resolution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter3D(x_arr, y_arr, z_arr, c=z_arr, cmap='jet', linewidths=0.5, label="Test Sphere Points")
        if stativ:
            ax.scatter3D(x_arr_zyl, y_arr_zyl, z_arr_zyl, c=z_arr_zyl, cmap='viridis', linewidths=0.5,
                     label="Test Zylinder Points")
        ax.set_title("3D Scatter Plot of the LTS Sphere")
        ax.set_xlabel("X Coorinates")
        ax.set_ylabel("Y Coorinates")
        ax.set_zlabel("Z Coorinates")
        ax.legend()

        plt.show()

    if stativ:
        x_data = numpy.hstack((x_arr, x_arr_zyl))
        y_data = numpy.hstack((y_arr, y_arr_zyl))
        z_data = numpy.hstack((z_arr, z_arr_zyl))
    else:
        x_data = x_arr
        y_data = y_arr
        z_data = z_arr

    return x_data, y_data, z_data

if __name__ == "__main__":



    # create TLS Data
    #tls_x, tls_y, tls_z = create_tls_data(sphere_x, sphere_y, sphere_z, sphere_radius)

    sphere_data_1 = numpy.loadtxt(open(r"data\SP1_Scan002_sphere.txt"), delimiter=",")

    sphere_x_data_1 = sphere_data_1[:, 0]
    sphere_x_data_2 = sphere_data_1[:, 1]
    sphere_x_data_3 = sphere_data_1[:, 2]




    sphere_data_max_z = sphere_data_1[sphere_data_1[:,2]==numpy.nanmax(sphere_data_1[:, 2])]
    print("sphere_data_mx_z: ", sphere_data_max_z)
    r_init = 0.15
    z_init_max = numpy.nanmax(sphere_data_1[:,2]) - r_init
    x_init = sphere_data_max_z[0,0] - r_init
    y_init = sphere_data_max_z[0,1] - r_init
    z_init = z_init_max - r_init
    v_init = numpy.zeros(4)

    print("Max  z koord: ", z_init_max)
    print("Mean x koord: ", x_init)
    print("Mean y koord: ", y_init)
    print("Z init", z_init)
    print("v init", v_init)



    w = (sphere_x_data_1)


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter3D(sphere_x_data_1, sphere_x_data_2, sphere_x_data_3, c=sphere_x_data_3, linewidths=0.5, label="LTS Sphere Points")
    ax.scatter3D([sphere_data_max_z[0,0]], [sphere_data_max_z[0,1]], [z_init_max], c=[z_init_max], linewidths=0.5, label="initial Center Point")
    ax.set_title("3D Scatter Plot of the LTS Sphere")
    ax.set_xlabel("X Coorinates")
    ax.set_ylabel("Y Coorinates")
    ax.set_zlabel("Z Coorinates")
    ax.legend()

    plt.show()




    print("Programm ENDE")