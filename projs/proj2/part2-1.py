# EE125 Project 2 - Part 2-1, Aliasing due to Undersampling
# Sept. 23, 2024

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# a.) plot signal x
t = [0, 0.5, 1]
x = [1, 3, -1.1]

# plt.plot(t, x, marker='o')
# plt.xlabel("Time (s)")
# plt.ylabel("x(t)")
# plt.title("Python Interpolation of Signal")
# plt.grid()
# plt.show()

# b.) plot h(n) as a stem plot
# h = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0]
# plt.stem(h, basefmt=" ")
# plt.xlabel('n')
# plt.ylabel('h(n)')
# plt.title("Impulse Response, Triangular-shaped")
# plt.grid()
# plt.show()

# c.) plot expanded signal x_e(n)
xe = [1, 0, 0, 0, 0, 3, 0, 0, 0, 0, -1.1] # original expanded x(n)
te = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

figure, (subplot1, subplot2) = plt.subplots(1, 2, figsize=(15,10))
# first subplot is expanded, but in samples and no t
subplot1.stem(xe, basefmt=" ")

subplot1.set_xlabel('n, in samples')
subplot1.set_ylabel('x_e(n)')
subplot1.set_title('Expanded Signal in Samples')
subplot1.grid(True)

# Second subplot is expanded, but in t (seconds) that starts at 0
subplot2.stem(te, xe, basefmt=" ")

subplot2.set_xlabel('t, in seconds')
subplot2.set_ylabel('x_e(nT)')
subplot2.set_title('Expanded Signal x_e(n) at Times to Interpolate')
subplot2.grid(True)

plt.tight_layout()
plt.show()

# d.) use conv to get x_e(n) * h(n),
h = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0]
xe = [1, 0, 0, 0, 0, 3, 0, 0, 0, 0, -1.1]
x_orig_from_conv = np.convolve(xe, h)
# plt.stem(x_orig_from_conv, basefmt=" ")
# plt.xlabel('n, time in samples')
# plt.ylabel('x_e(n) * h(n)')
# plt.title("Interpolated Signal from Convolution of x_e(n) and h(n)")
# plt.grid()
# plt.show()

# e.) replot the interpolated signal in seconds.
tn = np.linspace(0, 2, len(h) + len(xe) - 1)
# Need to fit convolution by shifting by .5 samples
# to produce fit that is same as x(n)
# plt.stem(np.subtract(tn, 0.5), x_orig_from_conv, basefmt=" ")
# plt.xlabel('t, time in seconds')
# plt.ylabel('x_e(n) * h(n)')
# plt.title("Interpolated Signal from Convolution of x_e(n) and h(n)")
# plt.grid()
# plt.show()

# f.) Discussion:
#       The approach for both d and e interpolates between original sample values.
#       This does extrapolate in time, as the sample space had the signal from 0 to 20.
#       However, as the original signal has the three key values at 0, 0.5, and 1,
#       some extrapolation must occur. In the original signal x(n), when n = -1 and -3
#       x(n) is not given. Therefore, the convolution approach seems to explicitly assume
#       at n = -1 and n = 3 (in seconds, not samples), that the convolution would use the value 0.

