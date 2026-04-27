import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import scipy
from scipy.special import ellipk, ellipkm1
from math import pi, sqrt

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

def v_rn(rho, z, M):
    return -np.log(1+M/sqrt(rho**2 + z**2))

def v_mp(rho, z, sigma, m):
    l2 = sqrt((rho + sigma)**2 + z**2)
    k2 = 4*sigma*rho/l2**2
    if k2 >= 0.95:
        K = ellipkm1(1-k2)
    else:
        K = ellipk(k2)
    return -np.log(1 + 2*m*K/(pi*l2))

def accessible_region_condition(v, rho, Lambda, u_rho_0, eps, l):
    grhorho = np.exp(2*(Lambda-v))
    if -1 + eps**2 * np.exp(-2 * v) - l**2 / rho**2 * np.exp(2*v) >= 0:
        u_rho_max = sqrt((-1 + eps**2 * np.exp(-2 * v) - l**2 / rho**2 * np.exp(2*v))/grhorho)
        if  u_rho_max >= u_rho_0:
            return True
        else:
            return False
    else:
        return False

def processing(
    data_path,
    rho_start,
    rho_end,
    u_rho_start,
    u_rho_end,
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
    samples,
):

    rho_start = rho_start + perturbation
    rho_end = rho_end + perturbation

    l = l*M
    b = b*M
    m = m*M
    sigma = (b*(b-2*M))**0.5
    z = z*M

    color = []

    rho_0 = []
    u_rho_0 = []

    destiny = []

    for i in range(samples):
        with open(f"{data_path}/ic_{i+1}.csv", mode="r") as file:
            csv_reader = csv.reader(file)

            for row in csv_reader:
                split_row = row[0].split(';')
                rho_0.append(float(split_row[0]))
                u_rho_0.append(float(split_row[1]))

    for i in range(number_of_points**2):

        if (i+1)%10000 == 0:
            ratio = (i+1)/number_of_points**2
            print(f"{ratio *100:.2f}% processed")

        with open(f"{data_path}/trajectory_{i}.csv", mode='r') as file:

            if os.path.getsize(f"{data_path}/trajectory_{i}.csv") == 0:
                rho_last = 0.0
                t_last = 0.0
                Lambda = 0.0

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
                elif rn_mp:
                    Lambda = 0.0
            
            if schw_bw:
                v_bh = v_schwarzschild(rho_0[i], z, sigma, M)
                v_ring = v_bachweyl(rho_0[i], z, sigma, m)
            elif rn_mp:
                v_bh = v_rn(rho_0[i], z, M)
                v_ring = v_mp(rho_0[i], z, sigma, m)

            v = v_bh + v_ring

            inside_the_region = accessible_region_condition(v, rho_0[i], Lambda, u_rho_0[i], eps, l)
            
            if inside_the_region == False:
                rho_last = "-"
                t_last = "-"

        COLOR_PLUNGE   = (122/255, 179/255, 239/255)  # light blue
        COLOR_ORBIT    = ( 22/255, 102/255, 186/255)  # dark blue
        COLOR_OUTSIDE  = (1.0,     1.0,     1.0    )  # white

        if type(rho_last) == float:
            if t_last < Tmax:
                color.append(COLOR_PLUNGE)
                destiny.append("plunge")
            elif t_last == Tmax:
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

    fig, ax = plt.subplots()
    ax.imshow(color_grid, origin='lower', aspect='auto',
            extent=[rho_start, rho_end, u_rho_start, u_rho_end])
    ax.set_xlabel(r'$\rho$ [M]')
    ax.set_ylabel(r'$u^\rho$ [1]')

    if schw_bw:
        name = f"basin_map_sch_bw_{number_of_points}_{M}_{l}_{eps}_{b}_{m}_{z}_{perturbation}.pdf"
    elif rn_mp:
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

    print(f"fractal dimension: {2-slope} ± {slope_error}") # 2 is the dimension of our phase space

# schw_bw_1.0_3.943_0.955_20_0.5_0.2

# destiny_unperturbed = processing(
#     data_path="./data_3",
#     rho_start = 11.65,
#     rho_end = 12,
#     u_rho_start = 0.077,
#     u_rho_end = 0.082,
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
#     samples = 5,
# )

# destiny_perturbed_1 = processing(
#     data_path="./data_5",
#     rho_start = 11.65,
#     rho_end = 12,
#     u_rho_start = 0.077,
#     u_rho_end = 0.082,
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
#     samples = 5,
# )

# destiny_perturbed_2 = processing(
#     data_path="./data_6",
#     rho_start = 11.65,
#     rho_end = 12,
#     u_rho_start = 0.077,
#     u_rho_end = 0.082,
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
#     samples = 5,
# )

# destiny_perturbed_3 = processing(
#     data_path="./data_7",
#     rho_start = 11.65,
#     rho_end = 12,
#     u_rho_start = 0.077,
#     u_rho_end = 0.082,
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
#     samples = 5,
# )

# destiny_perturbed_4 = processing(
#     data_path="./data_8",
#     rho_start = 11.65,
#     rho_end = 12,
#     u_rho_start = 0.077,
#     u_rho_end = 0.082,
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
#     samples = 5,
# )

# destiny_perturbed_5 = processing(
#     data_path="./data_9",
#     rho_start = 11.65,
#     rho_end = 12,
#     u_rho_start = 0.077,
#     u_rho_end = 0.082,
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
#     samples = 5,
# )

# destiny_perturbed_6 = processing(
#     data_path="./data_10",
#     rho_start = 11.65,
#     rho_end = 12,
#     u_rho_start = 0.077,
#     u_rho_end = 0.082,
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
#     samples = 5,
# )

# fbar_1 = fbar(destiny_perturbed_1, destiny_perturbed_2)
# fbar_2 = fbar(destiny_perturbed_3, destiny_perturbed_4)
# fbar_3 = fbar(destiny_perturbed_5, destiny_perturbed_6)

# perturbations = [10**(-4), 10**(-3), 10**(-2)]
# fbars = [fbar_1, fbar_2, fbar_3]

# fractal_dim(perturbations, fbars, "schw_bw_1.0_3.943_0.955_20_0.5_0.2.pdf")

# fractal dimension: 1.8610841597291847 ± 0.028723757498522188

# rn_mp_1.0_3.750_0.955_20_0.5_0.2

destiny_unperturbed = processing(
    data_path="./data_0",
    rho_start = 15,
    rho_end = 17,
    u_rho_start = 0.15,
    u_rho_end = 0.18,
    perturbation = 0,
    M = 1.0,
    l = 3.750,
    eps = 0.955,
    b = 20,
    m = 0.5,
    z = 0.2,
    Tmax = 10**4,
    schw_bw = False,
    rn_mp = True,
    number_of_points = 500,
    samples = 5,
)

destiny_perturbed_1 = processing(
    data_path="./data_1",
    rho_start = 15,
    rho_end = 17,
    u_rho_start = 0.15,
    u_rho_end = 0.18,
    perturbation = 10**(-4),
    M = 1.0,
    l = 3.750,
    eps = 0.955,
    b = 20,
    m = 0.5,
    z = 0.2,
    Tmax = 10**4,
    schw_bw = False,
    rn_mp = True,
    number_of_points = 500,
    samples = 5,
)

destiny_perturbed_2 = processing(
    data_path="./data_2",
    rho_start = 15,
    rho_end = 17,
    u_rho_start = 0.15,
    u_rho_end = 0.18,
    perturbation = -10**(-4),
    M = 1.0,
    l = 3.750,
    eps = 0.955,
    b = 20,
    m = 0.5,
    z = 0.2,
    Tmax = 10**4,
    schw_bw = False,
    rn_mp = True,
    number_of_points = 500,
    samples = 5,
)

destiny_perturbed_3 = processing(
    data_path="./data_3",
    rho_start = 15,
    rho_end = 17,
    u_rho_start = 0.15,
    u_rho_end = 0.18,
    perturbation = 10**(-3),
    M = 1.0,
    l = 3.750,
    eps = 0.955,
    b = 20,
    m = 0.5,
    z = 0.2,
    Tmax = 10**4,
    schw_bw = False,
    rn_mp = True,
    number_of_points = 500,
    samples = 5,
)

destiny_perturbed_4 = processing(
    data_path="./data_4",
    rho_start = 15,
    rho_end = 17,
    u_rho_start = 0.15,
    u_rho_end = 0.18,
    perturbation = -10**(-3),
    M = 1.0,
    l = 3.750,
    eps = 0.955,
    b = 20,
    m = 0.5,
    z = 0.2,
    Tmax = 10**4,
    schw_bw = False,
    rn_mp = True,
    number_of_points = 500,
    samples = 5,
)

destiny_perturbed_5 = processing(
    data_path="./data_5",
    rho_start = 15,
    rho_end = 17,
    u_rho_start = 0.15,
    u_rho_end = 0.18,
    perturbation = 10**(-2),
    M = 1.0,
    l = 3.750,
    eps = 0.955,
    b = 20,
    m = 0.5,
    z = 0.2,
    Tmax = 10**4,
    schw_bw = False,
    rn_mp = True,
    number_of_points = 500,
    samples = 5,
)

destiny_perturbed_6 = processing(
    data_path="./data_6",
    rho_start = 15,
    rho_end = 17,
    u_rho_start = 0.15,
    u_rho_end = 0.18,
    perturbation = -10**(-2),
    M = 1.0,
    l = 3.750,
    eps = 0.955,
    b = 20,
    m = 0.5,
    z = 0.2,
    Tmax = 10**4,
    schw_bw = False,
    rn_mp = True,
    number_of_points = 500,
    samples = 5,
)

fbar_1 = fbar(destiny_perturbed_1, destiny_perturbed_2)
fbar_2 = fbar(destiny_perturbed_3, destiny_perturbed_4)
fbar_3 = fbar(destiny_perturbed_5, destiny_perturbed_6)

perturbations = [10**(-4), 10**(-3), 10**(-2)]
fbars = [fbar_1, fbar_2, fbar_3]

fractal_dim(perturbations, fbars, "rn_mp_1.0_3.750_0.955_20_0.5_0.2.pdf")

# fractal dimension: 1.7734547747740814 ± 0.027473193370768003