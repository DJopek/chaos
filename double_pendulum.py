import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import pi
import random

# constants
l_1 = 0.1
l_2 = 0.5
m_1 = 0.8
m_2 = 0.3
g = 9.81

# Cauchy problem

# y = [theta_1, theta_2, theta_1_dot, theta_2_dot]
# y0 = [theta_1_0, theta_2_0, theta_1_dot_0, theta_2_dot_0]

# generating initial conditions for some energy E
theta_1_0 = []
theta_2_0 = []
theta_1_dot_0 = []
theta_2_dot_0 = []

Energy = [4, 5, 6, 7]
number_of_initial_conditions = 3

total_numbers_of_initial_conditions = number_of_initial_conditions*len(Energy)

def initial_conditions(E, number_of_initial_conditions):

    for i in range(number_of_initial_conditions):
        theta_1_0.append(0)
        theta_2_0.append(np.random.uniform(-pi/20, pi/20))
        theta_2_dot_0.append(0)
        bucket = []
        bucket.append(np.sqrt(2*(E+l_1*g*(m_1+m_2)+l_2*m_2*g*np.cos(theta_2_0[i])-(l_1+l_2)*(m_1+m_2)*g)/(l_1**2*(m_1+m_2))))
        bucket.append(-np.sqrt(2*(E+l_1*g*(m_1+m_2)+l_2*m_2*g*np.cos(theta_2_0[i])-(l_1+l_2)*(m_1+m_2)*g)/(l_1**2*(m_1+m_2))))
        theta_1_dot_0.append(bucket[random.randint(0,1)])

for i in range(len(Energy)):
    initial_conditions(Energy[i], number_of_initial_conditions)

def f(t,y):
    return [y[2], 
            y[3], 
            (-g*np.sin(y[0])*(2*m_1+m_2)-g*m_2*np.sin(y[0]-2*y[1])-2*m_2*np.sin(y[0]-y[1])*(y[2]**2*l_1*np.cos(y[0]-y[1])+l_2*y[3]**2))/(l_1*(2*m_1+m_2-m_2*np.cos(2*y[0]-2*y[1]))), 
            (2*np.sin(y[0]-y[1])*(np.cos(y[0])*g*(m_1+m_2)+l_1*(m_1+m_2)*y[2]**2+m_2*l_2*y[3]**2*np.cos(y[0]-y[1])))/(l_2*(2*m_1+m_2-m_2*np.cos(2*y[0]-2*y[1])))]

# energy of the system
def E(y_1, y_2, y_3, y_4):
    total_energy = 0.5*l_1**2*y_3**2*(m_1+m_2)+0.5*m_2*l_2**2*y_4**2+m_2*l_1*l_2*y_3*y_4*np.cos(y_1-y_2)-l_1*g*np.cos(y_1)*(m_1+m_2)-l_2*m_2*g*np.cos(y_2)+(l_1+l_2)*g*(m_1+m_2)
    return total_energy

# Poincaré section
def Poincare_section(total_numbers_of_initial_conditions):
    theta_2 = []
    theta_2_dot = []

    colors = []

    for i in range(total_numbers_of_initial_conditions):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for r in range(6)])
                    for s in range(1)]
        for j in range(number_of_points-1):
            if solutions[i].y[0][j] <= 0 and solutions[i].y[0][j+1] >= 0:
                colors.append(color[0])
                theta_2.append(solutions[i].y[1][j+1])
                theta_2_dot.append(solutions[i].y[3][j+1])

    for m in range(len(theta_2)):
        plt.scatter((theta_2[m]+np.pi)%(2 * np.pi) - np.pi, theta_2_dot[m], c=colors[m], s=0.1)

    plt.show()

# solving the equations of motion
solutions = []

error = 0.001

# defining time interval
t_span = [0,1000]
t = np.linspace(0, 1000, 10000)

for i in range(total_numbers_of_initial_conditions):
    y0 = [theta_1_0[i], theta_2_0[i], theta_1_dot_0[i], theta_2_dot_0[i]]
    system_energy = E(y0[0], y0[1], y0[2], y0[3])
    print(y0)
    print(system_energy)
    solutions.append(solve_ivp(f, t_span, y0, t_eval=t, rtol = 1e-12, atol = 1e-14))
    number_of_points = len(solutions[i].t)
    
    for j in range(number_of_points):
        solution_energy = E(solutions[i].y[0][j], solutions[i].y[1][j], solutions[i].y[2][j], solutions[i].y[3][j])
        if np.abs(solution_energy - system_energy) > error:
            sys.exit('Maximum energy drift of {} exceeded.'.format(error))

Poincare_section(total_numbers_of_initial_conditions)