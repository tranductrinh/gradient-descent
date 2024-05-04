import matplotlib.pyplot as plt
import numpy as np


def cal_y(x):
    return x ** 2 - 2 * x + 1


plt.rcParams["figure.autolayout"] = True
plt.title('Function y = x^2 - 2x + 1')
plt.xlabel('x')
plt.ylabel('y')
x = np.linspace(-7, 7, 100)
plt.plot(x, cal_y(x))


def cal_derivative(x):
    return 2 * x - 2


def gradient_descent(learning_rate, x0):
    x = [x0]
    for it in range(1000):
        x_new = x[-1] - learning_rate * cal_derivative(x[-1])
        if abs(cal_derivative(x_new)) < 1e-6:
            break
        x.append(x_new)
    return x, it


(x, number_of_iterations) = gradient_descent(0.1, -5)
print('After %d iterations, found min: x = %f, y = %f, ' % (number_of_iterations, x[-1], cal_y(x[-1])))

for it in x:
    print('x = %f, y = %f' % (it, cal_y(it)))


x = np.asarray(x)
plt.plot(x, cal_y(x), 'ro', markersize=7)
plt.show()
