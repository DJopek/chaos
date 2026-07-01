import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from scipy.special import ellipk, ellipkm1
from math import pi, sqrt

# output file structure for schw+bw (for rn+mp it's the same but without lambda)
# t | phi | rho | z | ut | uphi | urho | uz | lambda

# basin map creation

def v_schwarzschild(rho, z, sigma, M):
    d1 = sqrt(rho**2 + (z - M)**2)
    d2 = sqrt(rho**2 + (z + M)**2)
    return 0.5 * np.log((d1 + d2 - 2*M) / (d1 + d2 + 2*M))

def v_bachweyl(rho, z, sigma, m):
    k2 = 4*sigma*rho/(z**2 + (rho+sigma)**2)
    if k2 >= 0.95:
        K = ellipkm1(1-k2)
    else:
        K = ellipk(k2)
    return -2 * m * K / (pi * sqrt(z**2 + (rho+sigma)**2))

def Nmin1_RN(rho, z, M):
    return 1+M/sqrt(rho**2 + z**2)

def Nmin1_MP(rho, z, sigma, m):
    l2 = sqrt((rho + sigma)**2 + z**2)
    k2 = 4*sigma*rho/l2**2
    if k2 >= 0.95:
        K = ellipkm1(1-k2)
    else:
        K = ellipk(k2)
    return 1 + 2*m*K/(pi*l2)

def accessible_region_condition(eps, l, gtt, gphiphi, g_rhorho, urho_0):
    if -1 - eps**2 * gtt - l**2 * gphiphi >= 0:
        urho_max = sqrt((-1 - eps**2 * gtt - l**2 * gphiphi)/g_rhorho)
        if urho_max >= urho_0:
            return True
        else:
            return False
    else:
        return False

def processing(
    rho_start,
    rho_end,
    urho_start,
    urho_end,
    perturbation,
    M,
    l,
    eps,
    b, # Schwarzschild coordinates
    m,
    z,
    Tmax,
    schw_bw,
    rn_mp,
    number_of_points,
    n=1,
    zofrho = False,
    samples=5,
    data_path = "./data",
):

    rho_start = rho_start + perturbation
    rho_end = rho_end + perturbation

    l = l*M
    b = b*M
    m = m*M
    z = z*M

    if schw_bw:
        sigma = (b*(b-2*M))**0.5
    elif rn_mp:
        sigma = b - M

    color = []

    rho_0 = []
    urho_0 = []

    destiny = []

    # initial conditions
    for i in range(samples):
        with open(f"{data_path}/ic_{i+1}.csv", mode="r") as file:
            csv_reader = csv.reader(file)

            for row in csv_reader:
                split_row = row[0].split(';')
                rho_0.append(float(split_row[0]))
                urho_0.append(float(split_row[1]))

    if zofrho:
        zvals = np.sqrt(np.array(rho_0)*M)*n

    # lookup for Lambda for rho0
    if schw_bw:
        Lambda_lookup = []

        for i in range(number_of_points**2//samples):
            if i % (number_of_points//samples) == 0:
                with open(f"{data_path}/trajectory_{i}.csv", mode='r') as file:
                    if os.path.getsize(f"{data_path}/trajectory_{i}.csv") == 0:
                        Lambda_lookup.append("-")
                    else:
                        csv_reader = csv.reader(file)
                        rows = list(csv_reader)

                        if len(rows) < 2:
                            split_row_first = rows[0][0].split(";")
                            split_row_last = rows[0][0].split(";")
                        else:
                            split_row_first = rows[0][0].split(";")
                            split_row_last = rows[-1][0].split(";")

                        Lambda_lookup.append(float(split_row_first[-1]))

        Lambda_plot = []
        rho_0_plot = []

        for i in range(len(Lambda_lookup)):
            rho_vals = np.linspace(rho_start, rho_end, number_of_points)
            if type(Lambda_lookup[i]) == float:
                Lambda_plot.append(Lambda_lookup[i])
                rho_0_plot.append(rho_vals[i])

        plt.scatter(rho_0_plot, Lambda_plot, s=1)
        name = f"lambda_sch_bw_{number_of_points}_{M}_{l}_{eps}_{b}_{m}_{z}_{perturbation}.png"
        plt.xlabel(r'$\rho$ [M]')
        plt.ylabel(r"$\lambda$ [1]")
        plt.savefig(name, dpi=300)
        plt.close()

    # main loop
    for i in range(number_of_points**2):

        if (i+1)%10000 == 0:
            ratio = (i+1)/number_of_points**2
            print(f"{ratio *100:.2f}% processed")

        with open(f"{data_path}/trajectory_{i}.csv", mode='r') as file:

            if os.path.getsize(f"{data_path}/trajectory_{i}.csv") == 0:
                rho_last = 0.0
                t_last = 0.0
                if schw_bw:
                    j = (i%(number_of_points**2//samples))
                    n = j%(number_of_points//samples)
                    idx = (j-n)//(number_of_points//samples)
                    Lambda = Lambda_lookup[idx]

            else:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)

                if len(rows) < 2:
                    split_row_first = rows[0][0].split(";")
                    split_row_last = rows[0][0].split(";")
                else:
                    split_row_first = rows[0][0].split(";")
                    split_row_last = rows[-1][0].split(";")

                rho_last = float(split_row_last[2]) 
                t_last = float(split_row_last[0])
                if schw_bw:
                    Lambda = float(split_row_first[-1])
            
            if schw_bw:
                v_bh = v_schwarzschild(rho_0[i], z, sigma, M)
                v_ring = v_bachweyl(rho_0[i], z, sigma, m)
                v = v_bh + v_ring
                gtt = -np.exp(-2*v)
                gphiphi = 1/rho_0[i]**2 * np.exp(2*v)
                if type(Lambda) == float:
                    g_rhorho = np.exp(2*(Lambda-v))
                    inside_the_region = accessible_region_condition(eps, l, gtt, gphiphi, g_rhorho, urho_0[i])
                elif -1 - eps**2 * gtt - l**2 * gphiphi >= 0:
                    if rho_0[i] < sigma and rho_0[i] > 5:
                        Lambda = 100
                    else:
                        Lambda = -100
                    g_rhorho = np.exp(2*(Lambda-v))
                    inside_the_region = accessible_region_condition(eps, l, gtt, gphiphi, g_rhorho, urho_0[i])
                else:
                    inside_the_region = False

            elif rn_mp:
                if zofrho:
                    z = zvals[i]
                Nmin1_bh = Nmin1_RN(rho_0[i], z, M)
                Nmin1_ring = Nmin1_MP(rho_0[i], z, sigma, m)
                N = 1/(Nmin1_bh+Nmin1_ring-1)
                gtt = -1/N**2
                gphiphi = 1/rho_0[i]**2 * N**2
                g_rhorho = 1/N**2 
                inside_the_region = accessible_region_condition(eps, l, gtt, gphiphi, g_rhorho, urho_0[i])

            if inside_the_region == False:
                rho_last = "-"
                t_last = "-"

        COLOR_PLUNGE   = (122/255, 179/255, 239/255)  # light blue
        COLOR_ORBIT    = ( 22/255, 102/255, 186/255)  # dark blue
        COLOR_OUTSIDE  = (1.0,     1.0,     1.0    )  # white

        tolerance = 10 # this is in case timestamp was set to 1

        if type(rho_last) == float:
            if t_last < Tmax - tolerance:
                color.append(COLOR_PLUNGE)
                destiny.append("plunge")
            elif Tmax - tolerance <= t_last and t_last <= Tmax + tolerance:
                color.append(COLOR_ORBIT)
                destiny.append("orbit")
            else:
                color.append(COLOR_OUTSIDE)
        else:
            color.append(COLOR_OUTSIDE)

    # I had just simple point plotting to see some visual but then asked Claude to change it for pixel plot
    color_array = np.array(color, dtype=np.float32)

    parallelisation_division = int(number_of_points/samples)

    color_blocks = color_array.reshape(samples, number_of_points, parallelisation_division, 3)

    color_rho_urho = color_blocks.transpose(1, 0, 2, 3).reshape(number_of_points, number_of_points, 3)

    color_grid = color_rho_urho.transpose(1, 0, 2)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(color_grid, origin='lower', aspect='auto',
            extent=[rho_start, rho_end, urho_start, urho_end])
    ax.set_xlabel(r'$\rho$ [M]')
    ax.set_ylabel(r'$u^\rho$ [1]')

    if schw_bw:
        name = f"basin_map_sch_bw_{number_of_points}_{M}_{l}_{eps}_{b}_{m}_{z}_{perturbation}.pdf"
    elif rn_mp:
        if zofrho:
            name = f"basin_map_rn_mp_{number_of_points}_{M}_{l}_{eps}_{b}_{m}_{n}_{perturbation}.pdf"
        else:
            name = f"basin_map_rn_mp_{number_of_points}_{M}_{l}_{eps}_{b}_{m}_{z}_{perturbation}.pdf"

    plt.savefig(name, dpi=300)
    plt.close()

    return destiny

# uncertainty exponent and fractal dimension calculation 

def fbar(destiny_perturbed_plus, destiny_perturbed_minus, number_of_points=500):
    counter = 0
    for i in range(len(destiny_unperturbed)):
        if destiny_unperturbed[i] != destiny_perturbed_plus[i] or destiny_unperturbed[i] != destiny_perturbed_minus[i]:
            counter += 1
    fbar = counter/number_of_points**2
    return fbar

def fractal_dim(perturbations, fbars, name):
    p , cov = np.polyfit(np.log(perturbations), np.log(fbars), 1, cov=True)
    slope, intercept = p
    slope_error = np.sqrt(cov[0][0])

    x = np.linspace(min(perturbations),max(perturbations),1000)
    y = np.exp(intercept)*x**slope

    plt.scatter(perturbations, fbars, s=5)
    plt.plot(x,y, label="linear fit", color="red", linestyle="-")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\ln(\varepsilon)$")
    plt.ylabel(r"$\ln(\bar{f})$")
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()

    print(f"fractal dimension: {2-slope} ± {slope_error}") # 2 is the dimension of our phase space

def fractal_dims_plot(fractal_dims, parameters, errors, name, xlabel):
    plt.errorbar(parameters, fractal_dims, yerr=errors, fmt='o', capsize=3, markersize=3)
    plt.xlabel(xlabel)
    plt.ylabel(r"$d$")
    plt.savefig(name, dpi=300)

# example use
# schw_bw_1.0_3.943_0.955_20_0.5_0.2

# destiny_unperturbed = processing(
#     data_path="./data_0",
#     rho_start = 11.65,
#     rho_end = 12,
#     urho_start = 0.077,
#     urho_end = 0.082,
#     perturbation = 0.,
#     M = 1.0,
#     l = 3.943,
#     eps = 0.955,
#     b = 20,
#     m = 0.5,
#     z = 0.2,
#     Tmax = 10**4,
#     schw_bw = True,
#     rn_mp = False,
#     number_of_points = 500,
# )

# destiny_perturbed_1 = processing(
#     data_path="./data_1",
#     rho_start = 11.65,
#     rho_end = 12,
#     urho_start = 0.077,
#     urho_end = 0.082,
#     perturbation = 10**(-4),
#     M = 1.0,
#     l = 3.943,
#     eps = 0.955,
#     b = 20,
#     m = 0.5,
#     z = 0.2,
#     Tmax = 10**4,
#     schw_bw = True,
#     rn_mp = False,
#     number_of_points = 500,
# )

# destiny_perturbed_2 = processing(
#     data_path="./data_2",
#     rho_start = 11.65,
#     rho_end = 12,
#     urho_start = 0.077,
#     urho_end = 0.082,
#     perturbation = -10**(-4),
#     M = 1.0,
#     l = 3.943,
#     eps = 0.955,
#     b = 20,
#     m = 0.5,
#     z = 0.2,
#     Tmax = 10**4,
#     schw_bw = True,
#     rn_mp = False,
#     number_of_points = 500,
# )

# destiny_perturbed_3 = processing(
#     data_path="./data_3",
#     rho_start = 11.65,
#     rho_end = 12,
#     urho_start = 0.077,
#     urho_end = 0.082,
#     perturbation = 10**(-3),
#     M = 1.0,
#     l = 3.943,
#     eps = 0.955,
#     b = 20,
#     m = 0.5,
#     z = 0.2,
#     Tmax = 10**4,
#     schw_bw = True,
#     rn_mp = False,
#     number_of_points = 500,
# )

# destiny_perturbed_4 = processing(
#     data_path="./data_4",
#     rho_start = 11.65,
#     rho_end = 12,
#     urho_start = 0.077,
#     urho_end = 0.082,
#     perturbation = -10**(-3),
#     M = 1.0,
#     l = 3.943,
#     eps = 0.955,
#     b = 20,
#     m = 0.5,
#     z = 0.2,
#     Tmax = 10**4,
#     schw_bw = True,
#     rn_mp = False,
#     number_of_points = 500,
# )

# destiny_perturbed_5 = processing(
#     data_path="./data_5",
#     rho_start = 11.65,
#     rho_end = 12,
#     urho_start = 0.077,
#     urho_end = 0.082,
#     perturbation = 10**(-2),
#     M = 1.0,
#     l = 3.943,
#     eps = 0.955,
#     b = 20,
#     m = 0.5,
#     z = 0.2,
#     Tmax = 10**4,
#     schw_bw = True,
#     rn_mp = False,
#     number_of_points = 500,
# )

# destiny_perturbed_6 = processing(
#     data_path="./data_6",
#     rho_start = 11.65,
#     rho_end = 12,
#     urho_start = 0.077,
#     urho_end = 0.082,
#     perturbation = -10**(-2),
#     M = 1.0,
#     l = 3.943,
#     eps = 0.955,
#     b = 20,
#     m = 0.5,
#     z = 0.2,
#     Tmax = 10**4,
#     schw_bw = True,
#     rn_mp = False,
#     number_of_points = 500,
# )

# fbar_1 = fbar(destiny_perturbed_1, destiny_perturbed_2)
# fbar_2 = fbar(destiny_perturbed_3, destiny_perturbed_4)
# fbar_3 = fbar(destiny_perturbed_5, destiny_perturbed_6)

# perturbations = [10**(-4), 10**(-3), 10**(-2)]
# fbars = [fbar_1, fbar_2, fbar_3]

# fractal_dim(perturbations, fbars, "schw_bw_1.0_3.943_0.955_20_0.5_0.2.pdf")

# example use rn+mp

#destiny_unperturbed = processing(
#    data_path = "./data_0",
#    rho_start = 1,
#    rho_end = 2,
#    urho_start = 0.18,
#    urho_end = 0.20,
#    perturbation = 0,
#    M = 1.0,
#    l = 3.2,
#    eps = 0.995,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = False,
#    rn_mp = True,
#    number_of_points = 500,
#    n = 1/10,
#    zofrho = True,
#    samples = 50,
#)
#
#destiny_perturbed_1 = processing(
#    data_path = "./data_1",
#    rho_start = 1,
#    rho_end = 2,
#    urho_start = 0.18,
#    urho_end = 0.20,
#    perturbation = 10**(-4),
#    M = 1.0,
#    l = 3.2,
#    eps = 0.995,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = False,
#    rn_mp = True,
#    number_of_points = 500,
#    n = 1/10,
#    zofrho = True,
#    samples = 50,
#)
#
#destiny_perturbed_2 = processing(
#    data_path = "./data_2", 
#    rho_start = 1,
#    rho_end = 2,
#    urho_start = 0.18,
#    urho_end = 0.20,
#    perturbation = -10**(-4),
#    M = 1.0,
#    l = 3.2,
#    eps = 0.995,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = False,
#    rn_mp = True,
#    number_of_points = 500,
#    n = 1/10,
#    zofrho = True,
#    samples = 50,
#)
#
#destiny_perturbed_3 = processing(
#    data_path = "./data_3", 
#    rho_start = 1,
#    rho_end = 2,
#    urho_start = 0.18,
#    urho_end = 0.20,
#    perturbation = 10**(-3),
#    M = 1.0,
#    l = 3.2,
#    eps = 0.995,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = False,
#    rn_mp = True,
#    number_of_points = 500,
#    n = 1/10,
#    zofrho = True,
#    samples = 50,
#)
#
#destiny_perturbed_4 = processing(
#    data_path = "./data_4", 
#    rho_start = 1,
#    rho_end = 2,
#    urho_start = 0.18,
#    urho_end = 0.20,
#    perturbation = -10**(-3),
#    M = 1.0,
#    l = 3.2,
#    eps = 0.995,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = False,
#    rn_mp = True,
#    number_of_points = 500,
#    n = 1/10,
#    zofrho = True,
#    samples = 50,
#)
#
#destiny_perturbed_5 = processing(
#    data_path = "./data_5", 
#    rho_start = 1,
#    rho_end = 2,
#    urho_start = 0.18,
#    urho_end = 0.20,
#    perturbation = 10**(-2),
#    M = 1.0,
#    l = 3.2,
#    eps = 0.995,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = False,
#    rn_mp = True,
#    number_of_points = 500,
#    n = 1/10,
#    zofrho = True,
#    samples = 50,
#)
#
#destiny_perturbed_6 = processing(
#    data_path = "./data_6", 
#    rho_start = 1,
#    rho_end = 2,
#    urho_start = 0.18,
#    urho_end = 0.20,
#    perturbation = -10**(-2),
#    M = 1.0,
#    l = 3.2,
#    eps = 0.995,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = False,
#    rn_mp = True,
#    number_of_points = 500,
#    n = 1/10,
#    zofrho = True,
#    samples = 50,
#)
#
#fbar_1 = fbar(destiny_perturbed_1, destiny_perturbed_2)
#fbar_2 = fbar(destiny_perturbed_3, destiny_perturbed_4)
#fbar_3 = fbar(destiny_perturbed_5, destiny_perturbed_6)
#
#perturbations = [10**(-4), 10**(-3), 10**(-2)]
#fbars = [fbar_1, fbar_2, fbar_3]
#
#fractal_dim(perturbations, fbars, "rn_mp_1.0_3.2_0.995_20_0.5_0.1.pdf")

#fractal_dims_plot(
#    parameters = [0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995],
#    fractal_dims = [1.8127, 1.8103, 1.8087, 1.7493, 1.7647, 1.5836, 1.5257, 1.2424, 1.0788],
#    errors = [0.0162, 0.0014, 0.0039, 0.0025, 0.0015, 0.0016, 0.0236, 0.0266, 0.0356],
#    name = "d_eps_rn_mp",
#    xlabel = r"$\varepsilon$"
#)

destiny_unperturbed = processing(
    data_path = "./data_0",
    rho_start = 10,
    rho_end = 12,
    urho_start = 0.17,
    urho_end = 0.19,
    perturbation = 0,
    M = 1.0,
    l = 3.75,
    eps = 0.977,
    b = 20,
    m = 0.02, # m = [0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.3, 0.6, 1.1]
    z = 0.2,
    Tmax = 10**4,
    schw_bw = True,
    rn_mp = False,
    number_of_points = 500,
    samples = 100,
)

destiny_perturbed_1 = processing(
    data_path = "./data_1",
    rho_start = 10,
    rho_end = 12,
    urho_start = 0.17,
    urho_end = 0.19,
    perturbation = 10**(-4),
    M = 1.0,
    l = 3.75,
    eps = 0.977,
    b = 20,
    m = 0.02, # m = [0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.3, 0.6, 1.1]
    z = 0.2,
    Tmax = 10**4,
    schw_bw = True,
    rn_mp = False,
    number_of_points = 500,
    samples = 100,
)

destiny_perturbed_2 = processing(
    data_path = "./data_2",
    rho_start = 10,
    rho_end = 12,
    urho_start = 0.17,
    urho_end = 0.19,
    perturbation = -10**(-4),
    M = 1.0,
    l = 3.75,
    eps = 0.977,
    b = 20,
    m = 0.02, # m = [0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.3, 0.6, 1.1]
    z = 0.2,
    Tmax = 10**4,
    schw_bw = True,
    rn_mp = False,
    number_of_points = 500,
    samples = 100,
)

destiny_perturbed_3 = processing(
    data_path = "./data_3",
    rho_start = 10,
    rho_end = 12,
    urho_start = 0.17,
    urho_end = 0.19,
    perturbation = 10**(-3),
    M = 1.0,
    l = 3.75,
    eps = 0.977,
    b = 20,
    m = 0.02, # m = [0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.3, 0.6, 1.1]
    z = 0.2,
    Tmax = 10**4,
    schw_bw = True,
    rn_mp = False,
    number_of_points = 500,
    samples = 100,
)

destiny_perturbed_4 = processing(
    data_path = "./data_4",
    rho_start = 10,
    rho_end = 12,
    urho_start = 0.17,
    urho_end = 0.19,
    perturbation = -10**(-3),
    M = 1.0,
    l = 3.75,
    eps = 0.977,
    b = 20,
    m = 0.02, # m = [0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.3, 0.6, 1.1]
    z = 0.2,
    Tmax = 10**4,
    schw_bw = True,
    rn_mp = False,
    number_of_points = 500,
    samples = 100,
)

destiny_perturbed_5 = processing(
    data_path = "./data_5",
    rho_start = 10,
    rho_end = 12,
    urho_start = 0.17,
    urho_end = 0.19,
    perturbation = 10**(-2),
    M = 1.0,
    l = 3.75,
    eps = 0.977,
    b = 20,
    m = 0.02, # m = [0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.3, 0.6, 1.1]
    z = 0.2,
    Tmax = 10**4,
    schw_bw = True,
    rn_mp = False,
    number_of_points = 500,
    samples = 100,
)

destiny_perturbed_6 = processing(
    data_path = "./data_6",
    rho_start = 10,
    rho_end = 12,
    urho_start = 0.17,
    urho_end = 0.19,
    perturbation = -10**(-2),
    M = 1.0,
    l = 3.75,
    eps = 0.977,
    b = 20,
    m = 0.02, # m = [0.001, 0.003, 0.006, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.3, 0.6, 1.1]
    z = 0.2,
    Tmax = 10**4,
    schw_bw = True,
    rn_mp = False,
    number_of_points = 500,
    samples = 100,
)

fbar_1 = fbar(destiny_perturbed_1, destiny_perturbed_2)
fbar_2 = fbar(destiny_perturbed_3, destiny_perturbed_4)
fbar_3 = fbar(destiny_perturbed_5, destiny_perturbed_6)

perturbations = [10**(-4), 10**(-3), 10**(-2)]
fbars = [fbar_1, fbar_2, fbar_3]

fractal_dim(perturbations, fbars, "schw_bw_1.0_3.75_0.977_20_0.02_0.2.pdf")
