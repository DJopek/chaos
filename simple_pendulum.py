import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import pi
import random

# constants
l = 1
m = 1
g = 9.81

# Cauchy problem

# y = [theta, theta_dot]
# y0 = [theta_0, theta_dot_0]

#generating initial conditions
theta_0 = []
theta_dot_0 = []

for i in range(0,10):
    theta_0.append(np.random.uniform(-3, 3))
    theta_dot_0.append(np.random.uniform(-10, 10))

number_of_initial_conditions = len(theta_0)

def f(t,y):
    return [y[1], -g/l*np.sin(y[0])]

# defining time interval
t_span = [0,10]
t = np.linspace(0, 10, 10000)

# energy of the system
def E(y_1, y_2):
    total_energy = 0.5*m*l**2*y_2**2 + m*g*l*(1-np.cos(y_1))
    return total_energy

# solving the equations of motion
solutions = []

error = 0.1

for i in range(number_of_initial_conditions):
    y0 = [theta_0[i], theta_dot_0[i]]
    solutions.append(solve_ivp(f, t_span, y0, t_eval=t, rtol = 1e-6, atol = 1e-8))
    system_energy = E(y0[0], y0[1])
    number_of_points = len(solutions[i].t)

    for j in range(number_of_points):
        solution_energy = E(solutions[i].y[0][j], solutions[i].y[1][j])
        if np.abs(solution_energy - system_energy) > error:
            sys.exit('Maximum energy drift of {} exceeded.'.format(error))


# theta(t)
for i in range(number_of_initial_conditions):
    plt.plot(solutions[i].t, solutions[i].y[0])

plt.xlabel(r"$t\ [s]$")
plt.ylabel(r"$\theta_1\ [rad]$")
plt.show()

# phase space
for i in range(number_of_initial_conditions):
    # Wrap theta to the interval [-pi, pi] for a clearer plot
    wrapped_theta = (solutions[i].y[0] + np.pi) % (2 * np.pi) - np.pi
    plt.plot(wrapped_theta, solutions[i].y[1])

plt.xlabel(r"$\theta_1\ [rad]$")
plt.ylabel(r"$\dot{\theta_1}\ [rad \cdot s^{-1}]$")
plt.show()

# Poincaré section
def Poincare_section(total_numbers_of_initial_conditions):
    theta_1 = []
    theta_1_dot = []

    colors = []

    for i in range(total_numbers_of_initial_conditions):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for r in range(6)])
                    for s in range(1)]
        for j in range(number_of_points-1):
            if solutions[i].y[0][j] <= 0 and solutions[i].y[0][j+1] >= 0:
                colors.append(color[0])
                theta_1.append(solutions[i].y[0][j+1])
                theta_1_dot.append(solutions[i].y[1][j+1])
    
    for m in range(len(theta_1)):
        plt.scatter(theta_1[m], theta_1_dot[m], c=colors[m], s=10)

    plt.xlim(-0.5, 0.5)
    plt.xlabel(r"$\theta_1\ [rad]$")
    plt.ylabel(r"$\dot{\theta_1}\ [rad \cdot s^{-1}]$")
    plt.show()

Poincare_section(number_of_initial_conditions)