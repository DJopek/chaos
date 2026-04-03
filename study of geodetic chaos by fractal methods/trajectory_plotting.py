import csv
import matplotlib.pyplot as plt
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
from datashader.utils import export_image

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

plot("./data/test_0.csv")

# df = pd.DataFrame({'x': x, 'y': y})
# canvas = ds.Canvas(plot_width=800, plot_height=600)
# agg = canvas.points(df, 'x', 'y')
# img = tf.shade(agg)  # returns an image you can display or save

# export_image(img, "my_plot", export_path=".")  # saves as my_plot.png

# df = pd.DataFrame({'t': t, 'rho': rho})
# canvas = ds.Canvas(plot_width=800, plot_height=600)
# agg = canvas.points(df, 't', 'rho')
# img = tf.shade(agg)  # returns an image you can display or save

# export_image(img, "my_plot_t", export_path=".")  # saves as my_plot.png