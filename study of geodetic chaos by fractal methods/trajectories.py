import subprocess
from subprocess import Popen, PIPE
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import threading

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
    time_stamp,
    schw_bw,
    rn_mp,
    number_of_points,
    parallelisation_number=5, #number of points needs to be divisable by parallelisation number
    sample=int(input("sample: ")),
):

    parallelisation_division = int(number_of_points/parallelisation_number)

    rho = np.linspace(rho_start,rho_end, number_of_points)
    rho = rho + perturbation
    urho = np.linspace(urho_start,urho_end, number_of_points)
    urho = urho[(parallelisation_division*(sample-1)):(parallelisation_division*sample)]

    l = l*M
    b = b*M
    m = m*M
    z = z*M

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

calculate_trajectories(
    rho_start = 12,
    rho_end = 16,
    urho_start = 0.0,
    urho_end = 0.15,
    perturbation = 0,
    M = 1.0,
    l = 3.750,
    eps = 0.94,
    b = 15,
    m = 0.5,
    z = 0.0,
    Tmax = 10**5,
    time_stamp = 1000,
    schw_bw = False,
    rn_mp = True,
    number_of_points = 1000,
)