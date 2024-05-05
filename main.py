import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


def cal_y(x):
    return x ** 2 - 2 * x + 1


def cal_derivative(x):
    return 2 * x - 2


def gradient_descent(learning_rate, starting_point):
    x = [starting_point]
    for it in range(1000):
        x_new = x[-1] - learning_rate * cal_derivative(x[-1])
        if abs(cal_derivative(x_new)) < 1e-6:
            break
        x.append(x_new)
    return x, it


(x, number_of_iterations) = gradient_descent(0.25, -5)
print('After %d iterations, found min: x = %f, y = %f, ' % (number_of_iterations, x[-1], cal_y(x[-1])))
print('========= Details =========')

for it in x:
    print('x = %f, y = %f' % (it, cal_y(it)))

fig, ax = plt.subplots(1, 1)


def animate(i):
    ax.clear()
    plt.title('Iteration %d: x = %f, y = %f' % (i, x[i], cal_y(x[i])))
    plt.xlabel('x')
    plt.ylabel('y')
    x_org = np.linspace(-7, 9, 100)
    plt.plot(x_org, cal_y(x_org))
    if i > 0:
        ax.plot(x[i - 1], cal_y(x[i - 1]), color='blue', marker='o')
    ax.plot(x[i], cal_y(x[i]), color='red', marker='o')
    ax.set_xlim([-7, 9])
    ax.set_ylim([-5, 50])


animation = FuncAnimation(fig, animate, frames=len(x), interval=500, repeat=False)
animation.save("animation.gif", dpi=300, writer=PillowWriter(fps=1))
plt.close()
