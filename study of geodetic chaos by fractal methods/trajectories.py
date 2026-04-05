import subprocess
from subprocess import Popen, PIPE
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import threading

# rho_start = float(input("rho start in weyl coordinates: "))
# rho_end = float(input("rho end in weyl coordinates: "))
# u_rho_start = float(input("u_rho start in weyl coordinates: "))
# u_rho_end = float(input("u_rho end in weyl coordinates: "))
# number_of_points = int(input("number of points: "))
# parallelisation_number = int(input("number of parallel runs: "))
rho_start = 2
rho_end = 30
u_rho_start = 0.0
u_rho_end = 0.30
number_of_points = 1000
parallelisation_number = 5 #number of points needs to be divisable by parallelisation number
sample = int(input("sample: "))
parallelisation_division = int(number_of_points/parallelisation_number)

rho = np.linspace(rho_start,rho_end, number_of_points)
u_rho = np.linspace(u_rho_start,u_rho_end, number_of_points)
u_rho = u_rho[(parallelisation_division*(sample-1)):(parallelisation_division*sample)]

M = 1.0
l = 3.750*M
eps = 0.955
b = 20*M
m = 0.5*M
sigma = (b*(b-2*M))**0.5
z = 0.2*M
Tmax = 10**4
time_stemp = 100

process_0 = subprocess.Popen(["./Gravitacek2"], stdin=subprocess.PIPE, stdout=PIPE, stderr=PIPE, text=True)

response = process_0.stdout.readline()
print(response)

def drain_stderr(proc):
    for line in proc.stderr:
        pass

stderr_thread = threading.Thread(target=drain_stderr, args=(process_0,), daemon=True)
stderr_thread.start()

for i in range(len(rho)):
    for j in range(len(u_rho)):
        output_file = f"../data/test_{parallelisation_division*i+j+parallelisation_division*len(rho)*(sample-1)}.csv"
        print(f"trajectory_weyl(CombinedWeyl(WeylSchwarzschild({M}), BachWeylRing({m}, {sigma})), {eps}, {l}, {rho[i]*M}, {z},{u_rho[j]}, {Tmax}, {time_stemp}, {output_file})\n")
        process_0.stdin.write(f"trajectory_weyl(CombinedWeyl(WeylSchwarzschild({M}), BachWeylRing({m}, {sigma})), {eps}, {l}, {rho[i]*M}, {z}, {u_rho[j]}, {Tmax}, {time_stemp}, {output_file})\n")
        process_0.stdin.flush()
        while True:
            line = process_0.stdout.readline()
            print(line)
            if line.startswith("Time of execution"):
                break
        ic = [[rho[i],u_rho[j]]]
        with open(f'../ic_{sample}.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerows(ic)

process_0.stdin.close()
process_0.wait()