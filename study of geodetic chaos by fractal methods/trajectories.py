import subprocess
from subprocess import Popen, PIPE
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import threading
import sys

def calculate_trajectories(
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
    parallelisation_number=5, #number of points needs to be divisable by parallelisation number
    sample=int(sys.argv[1]),
):

    time_stamp = Tmax/10
    # for Poincaré sections
    # time_stamp = 1

    parallelisation_division = int(number_of_points/parallelisation_number)

    rho = np.linspace(rho_start,rho_end, number_of_points)
    rho = rho + perturbation
    urho = np.linspace(urho_start,urho_end, number_of_points)
    urho = urho[(parallelisation_division*(sample-1)):(parallelisation_division*sample)]

    l = l*M
    b = b*M
    m = m*M
    z = z*M

    if zofrho:
        zvals = np.sqrt(rho*M)*n

    process_0 = subprocess.Popen(["./Gravitacek2"], stdin=subprocess.PIPE, stdout=PIPE, stderr=PIPE, text=True)

    response = process_0.stdout.readline()
    print(response)

    # code for buffer handling was done with assistance of Claude
    def drain_stderr(proc):
        for line in proc.stderr:
            pass

    stderr_thread = threading.Thread(target=drain_stderr, args=(process_0,), daemon=True)
    stderr_thread.start()

    ic = []

    for i in range(len(rho)):
        for j in range(len(urho)):

            output_file = f"../data/trajectory_{parallelisation_division*i+j+parallelisation_division*len(rho)*(sample-1)}.csv"

            if zofrho:
                z = zvals[i]

            if schw_bw:
                sigma = (b*(b-2*M))**0.5 # b is in Schwarzschild coordinates, sigma is converted value to Weyl coordinates
                command = f"trajectory_weyl(CombinedWeyl(WeylSchwarzschild({M}), BachWeylRing({m}, {sigma})), {eps}, {l}, {rho[i]*M}, {z},{urho[j]}, {Tmax}, {time_stamp}, {output_file})\n"
            elif rn_mp:
                sigma = b-M # we want to have MP ring of the same size as we had with BW ring
                command = f"trajectory_mp(CombinedMP(ReissnerNordstrom({M}), MajumdarPapapetrouRing({m}, {sigma})), {eps}, {l}, {rho[i]*M}, {z},{urho[j]}, {Tmax}, {time_stamp}, {output_file})\n"

            print(command)
            process_0.stdin.write(command)
            process_0.stdin.flush()
            while True:
                line = process_0.stdout.readline()
                print(line)
                if line.startswith("Time of execution"):
                    break

            ic.append([rho[i],urho[j]])

    with open(f'../data/ic_{sample}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(ic)

    process_0.stdin.close()
    process_0.wait()

# example - basin map schw+bw

#calculate_trajectories(
#    rho_start = 11.65,
#    rho_end = 12,
#    urho_start = 0.077,
#    urho_end = 0.082,
#    perturbation = 0,
#    M = 1.0,
#    l = 3.943,
#    eps = 0.955,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = True,
#    rn_mp = False,
#    number_of_points = 500,
#    parallelisation_number = 5,
#)
#
#calculate_trajectories(
#    rho_start = 11.65,
#    rho_end = 12,
#    urho_start = 0.077,
#    urho_end = 0.082,
#    perturbation = 10**(-4),
#    M = 1.0,
#    l = 3.943,
#    eps = 0.955,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = True,
#    rn_mp = False,
#    number_of_points = 500,
#    parallelisation_number = 5,
#)
#
#calculate_trajectories(
#    rho_start = 11.65,
#    rho_end = 12,
#    urho_start = 0.077,
#    urho_end = 0.082,
#    perturbation = -10**(-4),
#    M = 1.0,
#    l = 3.943,
#    eps = 0.955,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = True,
#    rn_mp = False,
#    number_of_points = 500,
#    parallelisation_number = 5,
#)
#
#calculate_trajectories(
#    rho_start = 11.65,
#    rho_end = 12,
#    urho_start = 0.077,
#    urho_end = 0.082,
#    perturbation = 10**(-3),
#    M = 1.0,
#    l = 3.943,
#    eps = 0.955,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = True,
#    rn_mp = False,
#    number_of_points = 500,
#    parallelisation_number = 5,
#)
#
#calculate_trajectories(
#    rho_start = 11.65,
#    rho_end = 12,
#    urho_start = 0.077,
#    urho_end = 0.082,
#    perturbation = -10**(-3),
#    M = 1.0,
#    l = 3.943,
#    eps = 0.955,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = True,
#    rn_mp = False,
#    number_of_points = 500,
#    parallelisation_number = 5,
#)
#
#calculate_trajectories(
#    rho_start = 11.65,
#    rho_end = 12,
#    urho_start = 0.077,
#    urho_end = 0.082,
#    perturbation = 10**(-2),
#    M = 1.0,
#    l = 3.943,
#    eps = 0.955,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = True,
#    rn_mp = False,
#    number_of_points = 500,
#    parallelisation_number = 5,
#)
#
#calculate_trajectories(
#    rho_start = 11.65,
#    rho_end = 12,
#    urho_start = 0.077,
#    urho_end = 0.082,
#    perturbation = -10**(-2),
#    M = 1.0,
#    l = 3.943,
#    eps = 0.955,
#    b = 20,
#    m = 0.5,
#    z = 0.2,
#    Tmax = 10**4,
#    schw_bw = True,
#    rn_mp = False,
#    number_of_points = 500,
#    parallelisation_number = 5,
#)

# example - basin map rn+mp

#calculate_trajectories(
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
#    parallelisation_number = 50,
#)
#
#calculate_trajectories(
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
#    parallelisation_number = 50,
#)
#
#calculate_trajectories(
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
#    parallelisation_number = 50,
#)
#
#calculate_trajectories(
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
#    parallelisation_number = 50,
#)
#
#calculate_trajectories(
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
#    parallelisation_number = 50,
#)
#
#calculate_trajectories(
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
#    parallelisation_number = 50,
#)
#
#calculate_trajectories(
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
#    parallelisation_number = 50,
#)

# example - Poincaré section schw+bw

#calculate_trajectories(
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
#    parallelisation_number = 20,
#)

# example - Poincaré section rn+mp

#calculate_trajectories(
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
#    Tmax = 10**5,
#    schw_bw = False,
#    rn_mp = True,
#    number_of_points = 50,
#    n = 1/10,
#    zofrho = True,
#    parallelisation_number = 50,
#)
