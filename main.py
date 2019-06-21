
import numpy
import argparse
import matplotlib.pyplot as plt


from TLSToolbox import gaus_helmert_model, calc_sphere


if __name__ == "__main__":

    # commandline arguments
    parser = argparse.ArgumentParser("Gauss-Helmert-Ausgleich einer Kugel anhand von TLS Daten")
    parser.add_argument("number_of_epochs", type=int, help="number of how many measurements will be selected for the first estimation in the RANSAC")
    parser.add_argument("-v", "--verbosity", action="count", default=0,help="increase output verbosity")
    args = parser.parse_args()
    print("ARGS: ", args)

    # load TLS data
    sphere_data_1 = numpy.loadtxt(open(r"SP1_Scan002_sphere.txt"), delimiter=",")

    # select start values for sphere center - take the measuremnet where the zvalue is max
    sphere_data_max_z = sphere_data_1[sphere_data_1[:, 2] == numpy.nanmax(sphere_data_1[:, 2])]
    print("SPHERE DATA 1: ", sphere_data_1.shape)

    # indizes of the TLS data_block
    sphere_data_1_indizes = list(numpy.arange(0, sphere_data_1.shape[0], 1))

    # ab hier wir der



    # randomly select a specified number of TLS measurements
    sphere_data_1_indizes_rand_select = numpy.random.choice(sphere_data_1_indizes, args.number_of_epochs)

    print("\n# Random selected Indizes\n  ==========================")
    print(sphere_data_1_indizes_rand_select)

    # https://scipy-cookbook.readthedocs.io/items/Indexing.html
    sphere_data_1_subset = sphere_data_1[sphere_data_1_indizes_rand_select]

    # create figure for plotting
    fig = plt.figure()

    # calculate the parameters for the sphere out of the selected TLS data subset
    sphere_parameters, corrections, SIGMA_xx, sigma_0, fig, ax, status_det = gaus_helmert_model(sphere_data_1_subset, sphere_data_max_z,  fig, args.verbosity)

    # check if first estimation was correct
    if status_det:
        print("# Estimation was sucessfull")

        print("# Sig_0: ", sigma_0)

        # RANSAC-ING
        #d_i_n = numpy.sqrt((sphere_data_1_subset[:, 0] - sphere_parameters[0])**2 + (sphere_data_1_subset[:, 1] - sphere_parameters[1])**2 +(sphere_data_1_subset[:, 2] - sphere_parameters[2])**2) - sphere_parameters[3]
        print("# Check for in and outliers with full data set of shape: ", sphere_data_1.shape)
        d_i_n = abs(numpy.sqrt((sphere_data_1[:, 0] - sphere_parameters[0]) ** 2 + (sphere_data_1[:, 1] - sphere_parameters[1]) ** 2 + (sphere_data_1[:, 2] - sphere_parameters[2]) ** 2) - sphere_parameters[3])
        sigma_hersteller = 0.01     # laut hersteller angabe 10 mm ungenauigkeit

        print("\n# Bestimmung Inlier und Outlier"
              "\n  ==============================")

        print("# Hersteller Ungenauigkeit t: ", sigma_hersteller)
        print("# t^2: ", sigma_hersteller**2)
        print(d_i_n)

        in_and_outliers = numpy.where(d_i_n<sigma_hersteller, True, False)
        inliers = numpy.where(d_i_n<sigma_hersteller, 1, 0)
        outliers = numpy.where(d_i_n<sigma_hersteller, 0,1)

        print(in_and_outliers)
        print("# Number of Inliers: ", sum(inliers))
        print("# Number of Outliers: ", sum(outliers))
    else:
        print("# Estimation failed - determinante was lower than a thresholf")
        print("# returned x: ", sphere_parameters)

    # calculate a sphere visualisation for the estimates
    x_arr_sphere, y_arr_sphere, z_arr_sphere = calc_sphere(sphere_parameters[0], sphere_parameters[1], sphere_parameters[2], sphere_parameters[3], args.verbosity)

    # plot the resulting estimated sphere and starting parameters
    ax.plot_surface(x_arr_sphere, y_arr_sphere, z_arr_sphere, alpha=0.1, color='cyan', label="Estimated Sphere")
    ax.scatter3D([sphere_parameters[0]], [sphere_parameters[1]], [sphere_parameters[2]], color='black', linewidths=0.5, label="calculated Center Point")

    ax.set_title("3D Scatter Plot of the LTS Sphere")
    ax.set_xlabel("X Coorinates")
    ax.set_ylabel("Y Coorinates")
    ax.set_zlabel("Z Coorinates")
    ax.set_title("Estimate Sphere Coordinates and Sphere Parameters from TLS Measurements")

    plt.show()
    print("Programm ENDE")