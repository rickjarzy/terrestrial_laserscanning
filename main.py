
import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from numpy import dot


def calc_sphere(m_x, m_y, m_z, radius, verbose=False):


    phi = numpy.linspace(0, 2*numpy.pi, 100)
    theta = numpy.linspace(0, numpy.pi, 100)

    x = m_x + radius * numpy.outer(numpy.cos(phi), numpy.sin(theta))
    y = m_y + radius * numpy.outer(numpy.sin(phi), numpy.sin(theta))
    z = m_z + radius * numpy.outer(numpy.ones(numpy.size(phi)), numpy.cos(theta))

    if verbose:
        print(" len of x ", len(x))
        # 3d Scattrer plot - FULL resolution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, z, color='cyan', label="Test Sphere Points")

        plt.show()

    return x, y, z


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

    # load TLS data
    sphere_data_1 = numpy.loadtxt(open(r"data\SP1_Scan002_sphere.txt"), delimiter=",")

    # subset of sphere_data_1 to check if things work out
    sphere_data_1_indizes = list(numpy.arange(0,sphere_data_1.shape[0],1))

    sphere_data_1_indizes_rand_select = numpy.random.choice(sphere_data_1_indizes, 10)

    print(sphere_data_1_indizes_rand_select)

    # https://scipy-cookbook.readthedocs.io/items/Indexing.html
    sphere_data_1_subset = sphere_data_1[sphere_data_1_indizes_rand_select]
    sphere_x_data_1 = sphere_data_1_subset[:, 0]
    sphere_x_data_2 = sphere_data_1_subset[:, 1]
    sphere_x_data_3 = sphere_data_1_subset[:, 2]

    # select start values for sphere center - take the measuremnet where the zvalue is max
    sphere_data_max_z = sphere_data_1[sphere_data_1[:,2]==numpy.nanmax(sphere_data_1[:, 2])]

    # INITIAL VALUES
    # ===================================

    # estimate sphere radius
    r_c = 0.15
    # estimate sphere center coordinates
    z_init_max = sphere_data_max_z[0,2] - r_c
    z_c = z_init_max
    x_c = sphere_data_max_z[0,0] - r_c
    y_c = sphere_data_max_z[0,1] - r_c

    # initial values for the verbesserungen
    v = numpy.zeros(sphere_data_1_subset.shape[0]*3)

    print("Data diminesion: ", sphere_data_1.shape)
    print("Max  z koord: ", z_c)
    print("x koord init: ", x_c)
    print("y koord init: ", y_c)
    print("z koord init", z_c)
    print("r init: ", r_c)
    print("v init", v)

    # estimates for sphere koefficients
    x_0 = numpy.array([x_c, y_c, z_c, r_c])

    # write x,y,z measuerements onto easier readable variables
    x_1 = sphere_x_data_1
    y_1 = sphere_x_data_2
    z_1 = sphere_x_data_3

    #split verbesserungen onto the coordinate coefficients
    v_x_1 = v[0 : sphere_x_data_1.shape[0]]
    v_y_1 = v[sphere_x_data_1.shape[0] : sphere_x_data_1.shape[0]*2]
    v_z_1 = v[sphere_x_data_1.shape[0]*2 : sphere_x_data_1.shape[0]*3]

    print("v_x_1\n", v_x_1)
    print("v_y_1\n", v_y_1)
    print("v_z_1\n", v_z_1)

    # help value for the radius
    r_1 = numpy.sqrt((x_1+v_x_1-x_c)**2 + (y_1 + v_y_1 - y_c)**2 + (z_1 + v_z_1 - z_c)**2)

    # Kovarianzmatrix der Beobachtungen
    SIGMA = numpy.eye((sphere_data_1_subset.shape[0]*3))

    # Widerspruchsvektor
    w = (x_1 + v_x_1 - x_c)*(x_1 - x_c) + (y_1 + v_y_1 - y_c)*(y_1 - y_c) + (z_1 + v_z_1 - z_c)*(z_1 - z_c) - r_c*r_1

    # Designmatrix A
    A = -numpy.ones((sphere_data_1_subset.shape[0],4))

    A[:, 0] = -(x_1 + v_x_1 - x_c) / r_1
    A[:, 1] = -(y_1 + v_y_1 - y_c ) / r_1
    A[:, 2] = -(z_1 + v_z_1 - z_c ) / r_1

    # B is a sparse matrix - with the elements of A for each partial differential
    sparse_part_1 = numpy.eye(A.shape[0]) * A[:, 0] * -1
    sparse_part_2 = numpy.eye(A.shape[0]) * A[:, 1] * -1
    sparse_part_3 = numpy.eye(A.shape[0]) * A[:, 2] * -1

    # Bedingungsmatrix
    B = numpy.zeros((sparse_part_1.shape[0], sparse_part_1.shape[1]*3))
    B[:,0:sparse_part_1.shape[1]]=sparse_part_1
    B[:,sparse_part_1.shape[1]:sparse_part_1.shape[1]*2] = sparse_part_2
    B[:, sparse_part_1.shape[1]*2:sparse_part_1.shape[1] * 3] = sparse_part_3
    #numpy.savetxt("B_sparese.txt", B, delimiter=" ")


    print("Start calculation of the Normalequation matrix ...")

    # Normalgleichungsmatrix
    N = dot(dot(A.T,numpy.linalg.inv(dot(dot(B, SIGMA), B.T))),A)

    # Zuschläge
    x_d = -dot(numpy.linalg.inv(N),dot(dot(A.T,numpy.linalg.inv(dot(dot(B, SIGMA), B.T))),w))

    #Korrelat
    k = -dot(numpy.linalg.inv(dot(dot(B, SIGMA), B.T)),dot(A, x_d)+w)

    #Verbesserungen
    v = dot(dot(SIGMA, B.T), k)

    # kovarianz
    sigma_0 = numpy.sqrt(dot(dot(v.T,SIGMA), v)/(sphere_data_1_subset.shape[0]-sphere_data_1_subset.shape[1]))

    # kovarianzmatrix
    SIGMA_xx = N * sigma_0**2

    # verbesserten Parameter
    x_dach = x_0 + x_d

    print("x_0: ", x_0.shape)
    print(x_0)
    print("x_d: ", x_d.shape)
    print(x_d)
    print("x_dach: ", x_0 + x_d)
    print("Dim Check")
    print("A: ", A.shape)
    print("w: ", w.shape)
    print("r_1: ", r_1.shape)
    print("B: ", B.shape)
    print(B[:,1])
    print("N: ", N.shape)
    print(N)
    print("k: ", k.shape)
    print(k)
    print("v: ", v.shape)
    print(v)
    print("sigma_0: ", sigma_0.shape)
    print(sigma_0)
    print("SIGMA_xx: ", SIGMA_xx.shape)
    print(SIGMA_xx)



    x_arr_sphere, y_arr_sphere, z_arr_sphere = calc_sphere(x_dach[0], x_dach[1], x_dach[2], x_dach[3])


    # check if min
    # Minimumsforderung - erweitert um die Nebendedingung
    fl_check = dot(dot(v.T, numpy.linalg.inv(SIGMA)), v) - 2 * dot(k.T,(dot(A,x_d) + dot(B,v) + w))

    # hauptprobe - soll durchgeführt werden wenn die Minimumsforderung unter einem gewissen schwellwert gefallen ist
    f_l = numpy.sqrt((x_1 + v[0 : sphere_x_data_1.shape[0]] - x_dach[0])**2
                     + (y_1 + v[sphere_x_data_1.shape[0] : sphere_x_data_1.shape[0]*2] - x_dach[1])**2
                     + (z_1 + v[sphere_x_data_1.shape[0]*2 : sphere_x_data_1.shape[0]*3] - x_dach[2])**2
                     ) - x_dach[3]

    print("hauptbedingung: ", f_l)
    print("check bedingung: ", fl_check)
    print(f_l - fl_check)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x_arr_sphere, y_arr_sphere, z_arr_sphere, color='cyan', label="Estimated Sphere", )
    ax.scatter3D(sphere_x_data_1, sphere_x_data_2, sphere_x_data_3, c=sphere_x_data_3, linewidths=0.5, label="LTS Sphere Points")
    ax.scatter3D([sphere_data_max_z[0,0]], [sphere_data_max_z[0,1]], [z_init_max],  color='red', linewidths=0.5, label="initial Center Point")
    ax.set_title("3D Scatter Plot of the LTS Sphere")
    ax.set_xlabel("X Coorinates")
    ax.set_ylabel("Y Coorinates")
    ax.set_zlabel("Z Coorinates")


    plt.show()
    print("Programm ENDE")