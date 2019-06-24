import numpy


if __name__ == "__main__":

    # stanpunkt 1
    sphere_1_sp1 = numpy.array([-66820.7643,214342.8743,362.5963])
    sphere_2_sp1 = numpy.array([-66810.3776,214342.3883,362.8516])

    sphere_1_sp2 = numpy.array([-66820.777,214342.8643,362.5973])
    sphere_2_sp2 = numpy.array([-66810.39,214342.3823,362.8513])

    distance_sp1 = numpy.sqrt(sum((sphere_1_sp1 - sphere_2_sp1)**2))
    distance_sp2 = numpy.sqrt(sum((sphere_1_sp2 - sphere_2_sp2)**2))
    print("distance sp1: ", distance_sp1)
    print("distance sp1: ", distance_sp2)