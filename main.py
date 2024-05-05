import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# Change them and rerun
LEARNING_RATE = 0.25
STARTING_POINT = 8
MAX_NUMBER_OF_ITERATIONS = 1000
ALMOST_ZERO = 1e-6


# Calculate y from x
def cal_y(x):
    return x ** 2 - 2 * x + 1


# Calculate function derivative (or slope) at x
def cal_derivative(x):
    return 2 * x - 2


# Gradient descent
def gradient_descent(learning_rate, starting_point):
    x = [starting_point]
    for it in range(MAX_NUMBER_OF_ITERATIONS):
        x_new = x[-1] - learning_rate * cal_derivative(x[-1])
        if abs(cal_derivative(x_new)) < ALMOST_ZERO:
            break
        x.append(x_new)
    return x, it


(x, number_of_iterations) = gradient_descent(LEARNING_RATE, STARTING_POINT)
print('After %d iterations, found min at: x = %f, y = %f' % (number_of_iterations, x[-1], cal_y(x[-1])))
print('========= START Printing details =========')
for it in x:
    print('x = %f, y = %f' % (it, cal_y(it)))
print('========= END Printing details =========')

fig, ax = plt.subplots(1, 1)


# Function to render animation
def animate(i):
    ax.clear()
    plt.title('Simple Gradient Descent: y = x^2 - 2x + 1')
    x_org = np.linspace(-7, 9, 100)
    plt.plot(x_org, cal_y(x_org), label='Original function')
    if i > 0:
        ax.plot(x[i - 1], cal_y(x[i - 1]), color='blue', marker='o', label='Last x value')
        ax.annotate('',
                    xy=(((x[i] + x[i - 1]) / 2), cal_y(((x[i] + x[i - 1]) / 2))),
                    xytext=(x[i - 1], cal_y(x[i - 1])),
                    arrowprops={"width": 0.1, "headwidth": 5, 'headlength': 5, 'color': 'red'})

    ax.plot(x[i], cal_y(x[i]), color='red', marker='o', label='Current x value')
    ax.set_xlim([-7, 9])
    ax.set_ylim([-5, 50])
    plt.xlabel("Iteration #%d, x = %f, y = %f" % (i, x[i], cal_y(x[i])))
    plt.legend(loc="upper center")


# Save as animation.gif
print('========= START Animation =========')
animation = FuncAnimation(fig, animate, frames=len(x), interval=1000, repeat=False)
animation.save("animation.gif", dpi=300, writer=PillowWriter(fps=1))
plt.close()
print('========= DONE =========')
