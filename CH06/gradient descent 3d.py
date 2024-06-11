import numpy as np
import matplotlib.pyplot as plt


def z_function (x, y):
    F1 = 1.5 - 1.6*np.exp(-0.05*(3*np.power(x+3,2)+np.power(y+3,2)))
    F = F1 + (0.5 - np.exp(-0.1*(3*np.power(x-3,2)+np.power(y-3,2))))
    return F

def calculate_gradient(x, y, h=1e-5):
    dFx = (z_function(x + h, y) - z_function(x - h, y)) / (2 * h)
    dFy = (z_function(x, y + h) - z_function(x, y - h)) / (2 * h)
    return dFx, dFy

x = np.arange(-6,6+h,h)
y = np.copy(x)

X, Y = np.meshgrid(x, y)

Z = z_function(X, Y)

current_pos1 = (4, 0, z_function(4, 0))
current_pos2 = (0, -5, z_function(0, -5))
current_pos3 = (-5, 2, z_function(-5, 2))


learning_rate = 0.1

ax = plt.subplot(projection = "3d", computed_zorder = False)

for _ in range(1000):
    X_derivative, Y_derivative = calculate_gradient(current_pos1[0], current_pos1[1])
    X_new, Y_new = current_pos1[0] - learning_rate * X_derivative, current_pos1[1] - learning_rate * Y_derivative
    current_pos1 = X_new, Y_new, z_function(X_new, Y_new)

    X_derivative, Y_derivative = calculate_gradient(current_pos2[0], current_pos2[1])
    X_new, Y_new = current_pos2[0] - learning_rate * X_derivative, current_pos2[1] - learning_rate * Y_derivative
    current_pos2 = X_new, Y_new, z_function(X_new, Y_new)

    X_derivative, Y_derivative = calculate_gradient(current_pos3[0], current_pos3[1])
    X_new, Y_new = current_pos3[0] - learning_rate * X_derivative, current_pos3[1] - learning_rate * Y_derivative
    current_pos3 = X_new, Y_new, z_function(X_new, Y_new)

    ax.plot_surface(X,Y,Z, cmap = "viridis", zorder = 0)
    ax.scatter(current_pos1[0], current_pos1[1], current_pos1[2], color="magenta", zorder = 1)
    ax.scatter(current_pos2[0], current_pos2[1], current_pos2[2], color="green", zorder = 1)
    ax.scatter(current_pos3[0], current_pos3[1], current_pos3[2], color="cyan", zorder = 1)
    
    plt.pause(0.001)
    ax.clear()
