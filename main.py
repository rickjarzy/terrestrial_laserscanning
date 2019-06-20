
import numpy
import argparse
import matplotlib.pyplot as plt


from TLSToolbox import gaus_helmert_model, calc_sphere


if __name__ == "__main__":

    # commandline arguments
    parser = argparse.ArgumentParser("Gauss-Helmert-Ausgleich einer Kugel anhand von TLS Daten")
    parser.add_argument("number_of_epochs", type=int, help="number of how many measurements will be selected for the Ausgleich")
    parser.add_argument("-v", "--verbosity", action="count", default=0,help="increase output verbosity")
    args = parser.parse_args()
    print("ARGS: ", args)
    # create TLS Data
    #tls_x, tls_y, tls_z = create_tls_data(sphere_x, sphere_y, sphere_z, sphere_radius)
    print(globals())
    # load TLS data

    sphere_data_1 = numpy.loadtxt(open(r"SP1_Scan002_sphere.txt"), delimiter=",")

    # select start values for sphere center - take the measuremnet where the zvalue is max
    sphere_data_max_z = sphere_data_1[sphere_data_1[:, 2] == numpy.nanmax(sphere_data_1[:, 2])]
    print("SPHERE DATA 1: ", sphere_data_1.shape)

    # indizes of the TLS data_block
    sphere_data_1_indizes = list(numpy.arange(0, sphere_data_1.shape[0], 1))

    # randomly select a specified number of TLS measurements
    sphere_data_1_indizes_rand_select = numpy.random.choice(sphere_data_1_indizes, args.number_of_epochs)

    print("\n# Random selected Indizes\n  ==========================")
    print(sphere_data_1_indizes_rand_select)

    # https://scipy-cookbook.readthedocs.io/items/Indexing.html
    sphere_data_1_subset = sphere_data_1[sphere_data_1_indizes_rand_select]

    # create figure for plotting
    fig = plt.figure()

    # calculate the parameters for the sphere out of the selected TLS data subset
    sphere_parameters, corrections, SIGMA_xx, fig, ax = gaus_helmert_model(sphere_data_1_subset, sphere_data_max_z,  fig, args.verbosity)

    # RANSACING

    d_i_n = ((sphere_data_1_subset[:, 0] - sphere_parameters[0])**2 + (sphere_data_1_subset[:, 1] - sphere_parameters[1])**2 +(sphere_data_1_subset[:, 2] - sphere_parameters[2])**2) - sphere_parameters[3]
    sigma_hersteller = 0.01     # laut hersteller angabe 10 mm ungenauigkeit

    print("# Bestimmung Inlier und Outlier"
          "\n  ==============================")
    print(d_i_n)

    in_and_outliers = numpy.where(d_i_n<sigma_hersteller**2, True, False)
    print(in_and_outliers)

    # calculate a sphere visualisation for the estimates
    x_arr_sphere, y_arr_sphere, z_arr_sphere = calc_sphere(sphere_parameters[0], sphere_parameters[1], sphere_parameters[2], sphere_parameters[3], args.verbosity)

    # plot the resulting estimated sphere and starting parameters
    ax.plot_surface(x_arr_sphere, y_arr_sphere, z_arr_sphere, alpha=0.2, color='cyan', label="Estimated Sphere")
    ax.scatter3D([sphere_parameters[0]], [sphere_parameters[1]], [sphere_parameters[2]], color='black', linewidths=0.5, label="calculated Center Point")

    ax.set_title("3D Scatter Plot of the LTS Sphere")
    ax.set_xlabel("X Coorinates")
    ax.set_ylabel("Y Coorinates")
    ax.set_zlabel("Z Coorinates")
    ax.set_title("Estimate Sphere Coordinates and Sphere Parameters from TLS Measurements")

    plt.show()
    print("Programm ENDE")