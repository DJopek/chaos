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

# Energy = np.linspace(5,5.6,8) # [0,100] 1.
# Energy = np.linspace(5.6,6.2,8) # [0,300] 2.
# Energy = np.linspace(6.2,6.4,8) # [0,500] 3.
# Energy = np.linspace(6.4,6.8,8) # [0,1000] 4.
# Energy = np.linspace(5.9,6.3,8) # [0,700] 2-3.

# Energy = np.linspace(6.3,6.8,12) # [0,1200] 3-4.

# Energy = np.linspace(5.4,6.8,20) # [0,2000] 0.
# Energy = np.linspace(5.6, 6.4, 16) # [0,700] 2_3
# Energy = np.linspace(6.2,6.8,18) # [0,1200] 3_4.

Energy = [5.865]

number_of_initial_conditions = 1

total_numbers_of_initial_conditions = number_of_initial_conditions*len(Energy)

def initial_conditions(E, number_of_initial_conditions):

    for i in range(number_of_initial_conditions):
        theta_1_0.append(0)
        theta_2_0.append(0)
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

        number_of_points = len(solutions[i].t)

        for j in range(number_of_points-1):
            if solutions[i].y[0][j] <= 0 and solutions[i].y[0][j+1] >= 0:
                colors.append(color[0])
                theta_2.append(solutions[i].y[1][j+1])
                theta_2_dot.append(solutions[i].y[3][j+1])

    for m in range(len(theta_2)):
        plt.scatter((theta_2[m]+np.pi)%(2 * np.pi) - np.pi, theta_2_dot[m], c=colors[m], s=0.5)

    plt.xlabel(r"$\theta_2\ [rad]$")
    plt.ylabel(r"$\dot{\theta_2}\ [rad \cdot s^{-1}]$")
    plt.show()

# solving the equations of motion
solutions = []

error = 0.001

# defining time interval
t_span = [0,10]

for i in range(total_numbers_of_initial_conditions):
    y0 = [theta_1_0[i], theta_2_0[i], theta_1_dot_0[i], theta_2_dot_0[i]]
    system_energy = E(y0[0], y0[1], y0[2], y0[3])
    print(y0)
    print(system_energy)
    solutions.append(solve_ivp(f, t_span, y0, dense_output=True, rtol = 1e-12, atol = 1e-14))
    number_of_points = len(solutions[i].t)
    
    for j in range(number_of_points):
        solution_energy = E(solutions[i].y[0][j], solutions[i].y[1][j], solutions[i].y[2][j], solutions[i].y[3][j])
        if np.abs(solution_energy - system_energy) > error:
            sys.exit('Maximum energy drift of {} exceeded.'.format(error))

Poincare_section(total_numbers_of_initial_conditions)

x_1 = []
x_2 = []
y_1 = []
y_2 = []

for i in range(len(solutions[0].t)):
    x_1.append(l_1*np.sin(solutions[0].y[0][i]))
    x_2.append(x_1[i]+l_2*np.sin(solutions[0].y[1][i]))
    y_1.append(-l_1*np.cos(solutions[0].y[0][i]))
    y_2.append(y_1[i]-l_2*np.cos(solutions[0].y[1][i]))

plt.plot(solutions[0].y[0], solutions[0].y[1])
plt.xlabel(r"$\theta_1\ [rad]$")
plt.ylabel(r"$\theta_2\ [rad]$")
plt.show()

plt.plot(x_1, y_1)
plt.show()
plt.plot(x_2, y_2)
plt.show()

plt.plot(x_1, y_1)
plt.plot(x_2, y_2)
plt.show()