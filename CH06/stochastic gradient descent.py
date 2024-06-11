import numpy as np
import matplotlib.pyplot as plt

h = 0.1

def z_function(x, y):
    F1 = 1.5 - 1.6 * np.exp(-0.05 * (3 * np.power(x + 3, 2) + np.power(y + 3, 2)))
    F = F1 + (0.5 - np.exp(-0.1 * (3 * np.power(x - 3, 2) + np.power(y - 3, 2))))
    return F

def calculate_gradient(x, y, h=1e-5):
    dFx = (z_function(x + h, y) - z_function(x - h, y)) / (2 * h)
    dFy = (z_function(x, y + h) - z_function(x, y - h)) / (2 * h)
    return dFx, dFy

x = np.arange(-6, 6 + h, h)
y = np.copy(x)

X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)

initial_positions = [(4, 0), (0, -5), (-5, 2)]
current_positions = [(x, y, z_function(x, y)) for x, y in initial_positions]

learning_rate = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder = False)

for _ in range(1000):
    for idx, (current_x, current_y, current_z) in enumerate(current_positions):
        
        random_x = current_x + np.random.normal(0, h)
        random_y = current_y + np.random.normal(0, h)
        
        X_derivative, Y_derivative = calculate_gradient(random_x, random_y)
        X_new, Y_new = current_x - learning_rate * X_derivative, current_y - learning_rate * Y_derivative
        current_positions[idx] = (X_new, Y_new, z_function(X_new, Y_new))
    
    ax.plot_surface(X, Y, Z, cmap="viridis", zorder=0, alpha=0.6)
    for idx, (x, y, z) in enumerate(current_positions):
        colors = ["magenta", "green", "cyan"]
        ax.scatter(x, y, z, color=colors[idx], zorder=1)
    
    plt.pause(0.001)
    ax.clear()

plt.show()
