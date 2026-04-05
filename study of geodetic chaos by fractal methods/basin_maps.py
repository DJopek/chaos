import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import scipy
from scipy.special import ellipk, ellipkm1
from math import pi, sqrt

# rho_start = float(input("rho start in weyl coordinates: "))
# rho_end = float(input("rho end in weyl coordinates: "))
# u_rho_start = float(input("u_rho start in weyl coordinates: "))
# u_rho_end = float(input("u_rho end in weyl coordinates: "))
# number_of_points = int(input("number of points: "))
# error = float(input("error estimate: "))
# singularity = float(input("singularity: "))

rho_start = 2
rho_end = 30
u_rho_start = 0.0
u_rho_end = 0.30
number_of_points = 1000
error = 0.001
Tmax = 10**4
samples = 5

M = 1.0
l = 3.750*M
eps = 0.955
b = 20*M
m = 0.5*M
sigma = (b*(b-2*M))**0.5
z = 0.2

# t | phi | rho | z | ut | uphi | urho | uz | lambda

def v_schwarzschild(rho, z):
    d1 = sqrt(rho**2 + (z - M)**2)
    d2 = sqrt(rho**2 + (z + M)**2)
    return 0.5 * np.log((d1 + d2 - 2*M) / (d1 + d2 + 2*M))

def v_bachweyl(rho, z):
    k2 = 4*sigma*rho/(z**2 + (rho+sigma)**2)
    if k2 >= 0.95:
        K = ellipkm1(1-k2)
    else:
        K = ellipk(k2)
    return -2 * m * K / (pi * sqrt(z**2 + (rho+sigma)**2))

def accessible_region_condition(v, rho, Lambda, u_rho_0):
    grhorho = np.exp(2*(Lambda-v))
    if -1 + eps**2 * np.exp(-2 * v) - l**2 / rho**2 * np.exp(2*v) >= 0:
        u_rho_max = sqrt((-1 + eps**2 * np.exp(-2 * v) - l**2 / rho**2 * np.exp(2*v))/grhorho)
        if  u_rho_max >= u_rho_0:
            return True
        else:
            return False
    else:
        return False

color = []

rho_0 = []
u_rho_0 = []

for i in range(samples):
    with open(f"ic_{i+1}.csv", mode="r") as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            split_row = row[0].split(';')
            rho_0.append(float(split_row[0]))
            u_rho_0.append(float(split_row[1]))

for i in range(number_of_points**2):

    with open(f"./data/test_{i}.csv", mode='r') as file:

        if os.path.getsize(f"./data/test_{i}.csv") == 0:
            rho_last = 0.0
            t_last = 0.0
            z = 0.0
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
            Lambda = float(split_row_first[-1])
            ut = float(split_row_last[4])
            uphi = float(split_row_last[5])
            z_last = float(split_row_last[3])

        
        v_schw = v_schwarzschild(rho_0[i], z)
        v_bw = v_bachweyl(rho_0[i], z)

        v = v_schw + v_bw

        inside_the_region = accessible_region_condition(v, rho_0[i], Lambda, u_rho_0[i])
         
        if inside_the_region == False:
            rho_last = "-"
            t_last = "-"

    COLOR_PLUNGE   = (122/255, 179/255, 239/255)  # light blue
    COLOR_ORBIT    = ( 22/255, 102/255, 186/255)  # dark blue
    COLOR_OUTSIDE  = (1.0,     1.0,     1.0    )  # white

    if type(rho_last) == float:
        if t_last < Tmax:
            color.append(COLOR_PLUNGE)
        elif t_last == Tmax:
            color.append(COLOR_ORBIT)
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
plt.savefig(f"basin_map_sch_bw_{number_of_points}_{M}_{l}_{eps}_{b}_{m}_{z}.pdf", dpi=300)
plt.close()