import csv
import matplotlib.pyplot as plt
import numpy as np
import random

def plot(filepath):
    x = []
    y = []
    T = []
    Rho = []

    with open(filepath, mode='r') as file:
                    
        csv_reader = csv.reader(file)

        for row in csv_reader:

            row_split = row[0].split(';')
            rho = float(row_split[2])
            phi = float(row_split[1])
            t = float(row_split[0])
            Rho.append(rho)
            x.append(rho*np.cos(phi))
            y.append(rho*np.sin(phi))
            T.append(t)

    plt.scatter(x,y,s=1)
    plt.show()
    plt.scatter(T,Rho,s=1)
    plt.show()

# i = random.randint(0,999999)
# print(i)
# plot(f"./data/test_{i}.csv")
# plot(f"./build/test.csv")