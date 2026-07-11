import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from scipy.special import ellipk, ellipkm1
from math import pi, sqrt

# output file structure for schw+bw (for rn+mp it's the same but without lambda)
# t | phi | rho | z | ut | uphi | urho | uz | lambda

# Poincaré section creation

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

    rho_section = []
    urho_section = []

    # main loop
    for i in range(number_of_points**2):

        if (i+1)%100 == 0:
            ratio = (i+1)/number_of_points**2
            print(f"{ratio *100:.2f}% processed")

        with open(f"{data_path}/trajectory_{i}.csv", mode='r') as file:

            if os.path.getsize(f"{data_path}/trajectory_{i}.csv") != 0:

                csv_reader = csv.reader(file, delimiter=';')
                rows = list(csv_reader)

                for row_prev, row_next in zip(rows[:-1], rows[1:]):
                    z_prev = float(row_prev[3])
                    z_next = float(row_next[3])
                    uz_next = float(row_next[7])

                    if z_prev >= 0 and z_next < 0 and uz_next < 0:
                        rho_section.append(float(row_next[2]))
                        urho_section.append(float(row_next[6]))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(
        rho_section,
        urho_section,
        s=0.3,
        c="black",
        marker=".",
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlabel(r'$\rho$ [M]')
    ax.set_ylabel(r'$u^\rho$ [1]')
    ax.set_xlim(0, 65)
    ax.set_ylim(-0.35, 0.35)

    if schw_bw:
        name = f"Poincare_section_sch_bw_{number_of_points}_{M}_{l}_{eps}_{b}_{m}_{z}_{perturbation}.pdf"
    elif rn_mp:
        if zofrho:
            name = f"Poincare_section_rn_mp_{number_of_points}_{M}_{l}_{eps}_{b}_{m}_{n}_{perturbation}.pdf"
        else:
            name = f"Poincare_section_rn_mp_{number_of_points}_{M}_{l}_{eps}_{b}_{m}_{z}_{perturbation}.pdf"

    plt.savefig(name, dpi=300)
    plt.close()

# example - schw+bw

#poincare_section = processing(
#    data_path="./data_0",
#    rho_start = 7,
#    rho_end = 45,
#    urho_start = -0.2,
#    urho_end = 0.2,
#    perturbation = 0,
#    M = 1.0,
#    l = 3.75,
#    eps = 0.977,
#    b = 20,
#    m = 0.1,
#    z = 0.0,
#    Tmax = 2*10**5,
#    schw_bw = True,
#    rn_mp = False,
#    number_of_points = 20,
#    samples = 20,
#)

# example - rn+mp

#poincare_section = processing(
#    data_path="./data_0",
#    rho_start = 3,
#    rho_end = 30,
#    urho_start = -0.3,
#    urho_end = 0.3,
#    perturbation = 0,
#    M = 1.0,
#    l = 3.2,
#    eps = 0.955,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 2*10**5,
#    schw_bw = False,
#    rn_mp = True,
#    number_of_points = 20,
#    n = 1/10,
#    zofrho = True,
#    samples = 20,
#)
