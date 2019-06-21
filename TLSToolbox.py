import matplotlib.pyplot as plt
import numpy
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot

def calc_sphere(m_x, m_y, m_z, radius, verbose):


    phi = numpy.linspace(0, 2*numpy.pi, 100)
    theta = numpy.linspace(0, numpy.pi, 100)

    x = m_x + radius * numpy.outer(numpy.cos(phi), numpy.sin(theta))
    y = m_y + radius * numpy.outer(numpy.sin(phi), numpy.sin(theta))
    z = m_z + radius * numpy.outer(numpy.ones(numpy.size(phi)), numpy.cos(theta))

    if verbose==2:
        print("- len of x ", len(x))
        # 3d Scattrer plot - FULL resolution
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, z, color='cyan', alpha=0.5, label="Test Sphere Points")

        plt.show()

    return x, y, z

def gaus_helmert_model(sphere_data_1_subset, sphere_data_max_z, fig, verbose):

    # INITIAL VALUES
    # ===================================

    # estimate sphere radius
    r_c = 0.15
    # estimate sphere center coordinates
    z_init_max = sphere_data_max_z[0, 2] - r_c
    z_c = z_init_max
    x_c = sphere_data_max_z[0, 0]  # - r_c
    y_c = sphere_data_max_z[0, 1]  # - r_c

    # estimates for sphere koefficients
    x_0 = numpy.array([x_c, y_c, z_c, r_c])

    # initial values for the verbesserungen
    v = numpy.zeros(sphere_data_1_subset.shape[0] * 3)

    # get measurements for x, y z
    x_1 = sphere_data_1_subset[:, 0]
    y_1 = sphere_data_1_subset[:, 1]
    z_1 = sphere_data_1_subset[:, 2]

    # Konvergenzgrenze
    EPSILON = 1e-6

    print("\n\n- Start values\n"
          "  ===============")

    print("- Data diminesion subset: ", sphere_data_1_subset.shape)
    print("- Max  z koord: ", z_c)
    print("- x koord sphere init: ", x_c)
    print("- y koord sphere init: ", y_c)
    print("- z koord sphere init", z_c)
    print("- r init: ", r_c)
    print("- v init", v)

    ax = fig.gca(projection='3d')
    ax.scatter3D(x_1, y_1, z_1, c=z_1, linewidths=0.5, label="LTS Sphere Points")
    ax.scatter3D([sphere_data_max_z[0, 0]], [sphere_data_max_z[0, 1]], [z_init_max], color='red',
                 linewidths=0.5, label="initial Center Point")

    print("\n\n- Start sphere parameter estimation"
          "\n  =================================")

    fl_check = 1
    cou = 0
    x_d = numpy.array([1,1,1,1])

    while max(abs(x_d)) > EPSILON: #EPSILON < fl_check:

        # update x,y,z with the corrections of v - (update)

        x_1 = x_1 + v[0: x_1.shape[0]]
        y_1 = y_1 + v[x_1.shape[0]: x_1.shape[0] * 2]
        z_1 = z_1 + v[x_1.shape[0] * 2: x_1.shape[0] * 3]

        # split verbesserungen onto the coordinate coefficients
        v_x_1 = v[0: x_1.shape[0]]
        v_y_1 = v[x_1.shape[0]: x_1.shape[0] * 2]
        v_z_1 = v[x_1.shape[0] * 2: x_1.shape[0] * 3]

        if verbose==2:
            print("- Verbesserungen nach Koordinaten")
            print("- v_x_1: ", v_x_1)
            print("- v_y_1: ", v_y_1)
            print("- v_z_1: ", v_z_1)

        # help value for the radius
        r_1 = numpy.sqrt((x_1 + v_x_1 - x_0[0]) ** 2 + (y_1 + v_y_1 - x_0[1]) ** 2 + (z_1 + v_z_1 - x_0[2]) ** 2)

        # Kovarianzmatrix der Beobachtungen
        SIGMA = numpy.eye((sphere_data_1_subset.shape[0] * 3))

        # Widerspruchsvektor
        w = ((x_1 + v_x_1 - x_0[0]) * (x_1 - x_0[0]) + (y_1 + v_y_1 - x_0[1]) * (y_1 - x_0[1]) + (
                    z_1 + v_z_1 - x_0[2]) * (z_1 - x_0[2]) - x_0[3] * r_1) / r_1

        # Designmatrix A
        A = -numpy.ones((sphere_data_1_subset.shape[0], 4))

        A[:, 0] = -(x_1 + v_x_1 - x_0[0]) / r_1
        A[:, 1] = -(y_1 + v_y_1 - x_0[1]) / r_1
        A[:, 2] = -(z_1 + v_z_1 - x_0[2]) / r_1

        # B is a sparse matrix - with the elements of A for each partial differential
        sparse_part_1 = numpy.eye(A.shape[0]) * A[:, 0] * -1
        sparse_part_2 = numpy.eye(A.shape[0]) * A[:, 1] * -1
        sparse_part_3 = numpy.eye(A.shape[0]) * A[:, 2] * -1

        # Bedingungsmatrix
        B = numpy.zeros((sparse_part_1.shape[0], sparse_part_1.shape[1] * 3))
        B[:, 0:sparse_part_1.shape[1]] = sparse_part_1
        B[:, sparse_part_1.shape[1]:sparse_part_1.shape[1] * 2] = sparse_part_2
        B[:, sparse_part_1.shape[1] * 2:sparse_part_1.shape[1] * 3] = sparse_part_3
        # numpy.savetxt("B_sparese.txt", B, delimiter=" ")

        print("\n\n- Start calculation of the Normalequation matrix - iteration nr: %d\n"
              "  =====================================================================" % cou)

        # Normalgleichungsmatrix
        N = dot(dot(A.T, numpy.linalg.inv(dot(dot(B, SIGMA), B.T))), A)

        if abs(numpy.linalg.det(N)) > EPSILON:

            # Zuschläge
            x_d = -dot(numpy.linalg.inv(N), dot(dot(A.T, numpy.linalg.inv(dot(dot(B, SIGMA), B.T))), w))

            # Korrelat
            k = -dot(numpy.linalg.inv(dot(dot(B, SIGMA), B.T)), dot(A, x_d) + w)

            # Verbesserungen
            v = dot(dot(SIGMA, B.T), k)

            # Standardabweichung der Gewichtseinheit
            sigma_0 = numpy.sqrt(
                dot(dot(v.T, SIGMA), v) / (sphere_data_1_subset.shape[0] - 4))

            # kovarianzmatrix der ausgeglichenen Kugelparameter
            SIGMA_xx = N * sigma_0 ** 2

            # verbesserten Parameter
            x = x_0 + x_d

            print("- x_0: ", x_0.shape, " - ", x_0)
            print("- x_d: ", x_d.shape, " - ", x_d)
            print("- x: ", x.shape, " - ", x)

            if verbose==2:
                print("\n- Dim Check A: ", A.shape)
                print("- Dim Check w: ", w.shape)
                print("- Dim Check r_1: ", r_1.shape)
                print("- Dim Check B: ", B.shape)
                print("- Dim Check N: ", N.shape)
                print(N)
                print("\n- N det: ", numpy.linalg.det(N))
                print("- abs(det(N)) < EPSILON: ", abs(numpy.linalg.det(N)) < EPSILON)
                print("\n- Korrelaten k: ", k.shape, " - ", k)

                print("\n- Verbesserungen v: ", v.shape, " - ", v)

                print("\n- sigma_0: ", sigma_0.shape)
                print(sigma_0)
                print("\n- SIGMA_xx: ", SIGMA_xx.shape)
                print(SIGMA_xx)

            # check if min
            # Minimumsforderung - erweitert um die Nebenbedingung
            fl_hauptprobe = dot(dot(v.T, numpy.linalg.inv(SIGMA)), v) - 2 * dot(k.T, (dot(A, x_d) + dot(B, v) + w))

            # hauptprobe - soll durchgeführt werden wenn die Minimumsforderung unter einem gewissen schwellwert gefallen ist
            f_l = numpy.sqrt((x_1 + v[0: x_1.shape[0]] - x[0]) ** 2
                             + (y_1 + v[x_1.shape[0]: x_1.shape[0] * 2] - x[1]) ** 2
                             + (z_1 + v[x_1.shape[0] * 2: x_1.shape[0] * 3] - x[2]) ** 2
                             ) - x[3]

            print("\n\n-  Kontrollen\n"
                  "   ==============")
            if verbose == 1:
                print("- Hauptbedingung: ", f_l)
                print("- max(abs(x_d)): ", max(abs(x_d)))
                print("- max(abs(x_d)) < EPSILON: ", max(abs(x_d)) < EPSILON)

            print("\n- Minimumsforderung: ", fl_hauptprobe)

            # Update iterative variables
            x_0 = x
            # v gets updated on the beginning of the while loop

        else:
            print("\n!! Normal equation Matrix is not invertable - No solution as found"
                  "\n   ==============================================================")

            x = numpy.empty([4])
            break

        # check if convergenze is smaller than EPSILON
        if fl_hauptprobe < EPSILON:

            print("\n- Hauptprobe erfolgreich - Konvergenz erreicht"
                  "\n  =============================================")
            print("- x_d max(abs()): ", max(abs(x_d)))
            print("- fl_hauptprobe: ", fl_hauptprobe)
            cou += 1
            #break
        # elif cou == 3:
        #    break
        else:
            cou += 1
            continue

    return x, v, SIGMA_xx, sigma_0, fig, ax