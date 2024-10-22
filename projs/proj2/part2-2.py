# EE125 Project 2 - Part 2-2, Sinc Interpolation
# Sept. 23, 2024

import numpy as np
import matplotlib.pyplot as plt

# a.) set up time vector
t = np.arange(-2, 2.1, 0.1)

# b.) write function sincInterp implementing eq 4
def sincInterp(x, T, t):
    """
    Performs Sinc Interpolation without default sinc functions.
    :param x:   Signal x(n), to interpolate. Should be a numpy array.
    :param T:   Sampling interval as integer, that was used to generate x(n) = x(nT)
    :param t:   numpy array for time vector
    :return:    numpy array for y(n), sinc interpolation of x(n)
    """
    y = np.zeros(len(t))
    # extracts time_index (0 to len(t)) and time (value from t)
    for time_index, time in enumerate(t):
        for n, xn in enumerate(x):
            # to avoid division over 0
            if (time - (n * T)) == 0:
                y[time_index] += xn
            else:
                y[time_index] += xn * np.sin(np.pi * (time - n * T) / T) / (np.pi * (time - n * T) / T)

    return y

# c.) compute h(n) for sinc interp function.
# [1] is the impulse
# Note, T must be 0.5 here, as our sampling interval is the same is it was in
# the previous part.
# plt.plot(t, sincInterp([1], 0.5, t), '-o')
# plt.xlabel('t, time in seconds')
# plt.ylabel('h(t)')
# plt.title("Impulse Reponse of Sinc Interpolation, h(n)")
# plt.grid()
# plt.show()

# d.) sinc-interpolate the original x(n)
#       Then, plot the sinc-interpolate, x(n), and the convolution interpolate together

T = 0.5 # Sampling Interval, from previous part

x = [1, 3, -1.1] # original x(n)
t_original = [0, 0.5, 1]

xe = [1, 0, 0, 0, 0, 3, 0, 0, 0, 0, -1.1] # extended signal
h = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0] # triangular signal
x_orig_from_conv = np.convolve(xe, h) # interpolated signal via convolution
tn = np.subtract(np.linspace(0, 2, len(h) + len(xe) - 1), T) # shifted time space for conv

plt.plot(t, sincInterp(x, 0.5, t), '-o', markersize='5', lw=1)
plt.plot(tn, x_orig_from_conv, marker="^", markersize='5', lw=1)
plt.plot(t_original, x, marker="1", markersize='10', ls='', color='black')
plt.xlabel('t, time in seconds')
plt.ylabel('Output y(t) and x(n)')
plt.title("Various Interpolations of x(n) and Original x(n)")
plt.legend(['Sinc Interpolation of x(n)', 'Linear Interpolation of x(n)', 'x(n)'])
plt.grid()
plt.show()