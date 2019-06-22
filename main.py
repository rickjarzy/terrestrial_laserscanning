
import numpy
import argparse
import matplotlib.pyplot as plt
import time

from TLSToolbox import gaus_helmert_model, calc_sphere


if __name__ == "__main__":

    start = time.perf_counter()
    # commandline arguments
    parser = argparse.ArgumentParser("Gauss-Helmert-Ausgleich einer Kugel anhand von TLS Daten")
    parser.add_argument("number_of_epochs", type=int, help="number of how many measurements will be selected for the first estimation in the RANSAC")
    parser.add_argument("-v", "--verbosity", action="count", default=0,help="increase output verbosity")
    args = parser.parse_args()

    tls_file_name_list = [r"SP1_Scan002_sphere.txt", r"SP1_Scan003_sphere.txt", r"SP2_Scan002_sphere.txt", r"SP2_Scan003_sphere.txt"]
    used_file = tls_file_name_list[0]

    # load TLS data

    sphere_data_1 = numpy.loadtxt(open(used_file), delimiter=",")


    # select start values for sphere center - take the measuremnet where the zvalue is max
    sphere_data_max_z = sphere_data_1[sphere_data_1[:, 2] == numpy.nanmax(sphere_data_1[:, 2])]
    print("SPHERE DATA available: ", sphere_data_1.shape)

    # indizes of the TLS data_block
    sphere_data_1_indizes = numpy.array(list(numpy.arange(0, sphere_data_1.shape[0], 1)))

    # probability for RANSAC iterations
    p = 0.99

    # start value for max inliers and the Number of Iterations in the RANSAC loop
    max_inliers = 0
    N = numpy.inf
    counter_ransac_loop = 0

    # figure for result plotting
    fig = plt.figure()

    # RANSAC-ING
    while N >= counter_ransac_loop:
        # randomly select a specified number of TLS measurements
        sphere_data_1_indizes_rand_select = numpy.random.choice(sphere_data_1_indizes, args.number_of_epochs)

        if args.verbosity == 2:
            print("\n# Random selected Indizes\n  ==========================")
            print(sphere_data_1_indizes_rand_select)

        # https://scipy-cookbook.readthedocs.io/items/Indexing.html
        sphere_data_1_subset = sphere_data_1[sphere_data_1_indizes_rand_select]

        # calculate the parameters for the sphere out of the selected TLS data subset
        sphere_parameters, corrections, SIGMA_xx, sigma_0, fig, ax, status_det = gaus_helmert_model(sphere_data_1_subset, sphere_data_max_z,  fig, args.verbosity)

        # check if first estimation was correct - this status should be true
        if status_det:

            #d_i_n = numpy.sqrt((sphere_data_1_subset[:, 0] - sphere_parameters[0])**2 + (sphere_data_1_subset[:, 1] - sphere_parameters[1])**2 +(sphere_data_1_subset[:, 2] - sphere_parameters[2])**2) - sphere_parameters[3]

            d_i_n = abs(numpy.sqrt((sphere_data_1[:, 0] - sphere_parameters[0]) ** 2 + (sphere_data_1[:, 1] - sphere_parameters[1]) ** 2 + (sphere_data_1[:, 2] - sphere_parameters[2]) ** 2) - sphere_parameters[3])
            sigma_hersteller = 0.01     # laut hersteller angabe 10 mm ungenauigkeit

            # used to select only the inliers of the entire data set
            in_and_outliers = numpy.where(d_i_n<sigma_hersteller, True, False)

            sum_inliers = sum(numpy.where(d_i_n < sigma_hersteller, 1, 0))
            sum_outliers = sum(numpy.where(d_i_n < sigma_hersteller, 0, 1))


            # print statements depending on verbosity level
            if args.verbosity == 1: print("\n# Estimation was sucessfull")

            if args.verbosity == 2:
                print("# Check for in and outliers with full data set of shape: ", sphere_data_1.shape)
                print("\n# Bestimmung Inlier und Outlier"
                      "\n  ==============================")
                print("# Hersteller Ungenauigkeit t: ", sigma_hersteller)
                print("# t^2: ", sigma_hersteller ** 2)

            # check if an update of the number of iterations in the while loop can be executed
            if sum_inliers > max_inliers:

                print("# UPDATE N - old num inliers %d < %d new num inliers " % (max_inliers, sum_inliers))

                max_inliers = sum_inliers

                eps = sum_outliers/sphere_data_1.shape[0]
                s = args.number_of_epochs
                N = int(round(numpy.log10(1-p) / numpy.log10(1 - (1 - eps)**s)))

                consensus_set = sphere_data_1[in_and_outliers]
                best_sphere_estimate = sphere_parameters

                # reset counter to walk again from the beginning
                #counter_ransac_loop = 0

                if args.verbosity == 2:
                    print(in_and_outliers)
                    print("# Number of Inliers: ", sum_inliers)
                    print("# Number of Outliers: ", sum_outliers)
                    print("# Number of inliers in the consensus set: ", consensus_set.shape)
                    print("# New Estimation of Iterations N: ", N)

            # because det was sucessfully dertermined - count the while loop iteration
            counter_ransac_loop += 1
        else:

            if args.verbosity == 1:
                # because the det was not successfully determined - this while loop iteration wont be counted
                print("\n# Estimation failed - determinante was lower than the set threshold")
            if args.verbosity == 2:
                print("# returned x: ", sphere_parameters)

            continue

    print("\n# RANSAC successfully finished in %d [sec]" % (time.perf_counter() - start),
          "\n  %s" % used_file,
          "\n  ============================================")
    print("# max inliers found: ", max_inliers)

    print("# Best sphere estimates  with found max inliers ")
    print("# Sphere X:  ", best_sphere_estimate[0])
    print("# Sphere Y:  ", best_sphere_estimate[1])
    print("# Sphere Z:  ", best_sphere_estimate[2])
    print("# Sphere R:  ", best_sphere_estimate[3])
    print("# Sphere Volume: ", (4/3) * numpy.pi * best_sphere_estimate[3]**3, "[m³]")

    #sphere_parameters, corrections, SIGMA_xx, sigma_0, fig, ax, status_det = gaus_helmert_model(consensus_set, sphere_data_max_z, fig, args.verbosity)

    # calculate a sphere visualisation for the estimates
    x_arr_sphere, y_arr_sphere, z_arr_sphere = calc_sphere(best_sphere_estimate[0], best_sphere_estimate[1], best_sphere_estimate[2], best_sphere_estimate[3], args.verbosity)

    # plot the resulting estimated sphere and starting parameters
    ax.plot_surface(x_arr_sphere, y_arr_sphere, z_arr_sphere, alpha=0.1, color='cyan', label="Estimated Sphere")
    ax.scatter3D([best_sphere_estimate[0]], [best_sphere_estimate[1]], [best_sphere_estimate[2]], color='black', linewidths=0.5, label="calculated Center Point")

    ax.set_title("3D Scatter Plot of the LTS Sphere")
    ax.set_xlabel("X Coorinates")
    ax.set_ylabel("Y Coorinates")
    ax.set_zlabel("Z Coorinates")
    ax.set_title("Estimate Sphere Coordinates and Sphere Parameters from TLS Measurements")

    plt.show()

    print("Programm ENDE")