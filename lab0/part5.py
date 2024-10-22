# EE125 Project 1 - Part 5, Moving Average and Convolution
# Sept. 16, 2024

import numpy as np
import scipy
import matplotlib.pyplot as plt


# 1. Implement movingAverage(), a three point moving average equation
# x is a vector with N elements
# b1, b2, b3 are float weights
# xm1 and xm2 are x[-1] and x[-2] respectively
# movingAvg returns a vector with N elements as well
def movingAvg(x, b1, b2, b3, xm1, xm2):
    N = len(x)
    y = np.zeros(N) # Assuming x is a numpy array, zero out all elements

    # For initial cases, n = 0, use xm1 = x[-1] and xm2 = x[-2]
    y[0] = b1 * x[0] + b2 * xm1 + b3 * xm2

    if N > 1:
        y[1] = b1 * x[1] + b2 * x[0] + b3 * xm1

    for i in range(2, N):
        y[i] = b1 * x[i] + b2 * x[i - 1] + b3 * x[i - 2]

    return y

# Validate that code is working correctly
# y_filt = scipy.signal.lfilter([b1,b2,b3],1,x,zi=zi)

print(f"Test 1: {movingAvg([1, 1, 1, 0, 0], 1, 1, 1, 0, 0)}")
# Test 1: [1. 2. 3. 2. 1.]

print(f"Test 2: {movingAvg([1, 1, 1, 0, 0], 1, 1, 1, 1, 1)}")
# Test 2: [3. 3. 3. 2. 1.]

print(f"Test 3: {movingAvg([1, 2, 3, 0, 0], 4, 5, 6, 7, 8)}")
# Test 3: [87. 55. 28. 27. 18.]
# All tests seem to follow the example solutions!

# 2. Pick some arbitrary b1, b2, b3, and compute the impulse response using movingAvg
N_2 = 10
dirac_delta = np.zeros(N_2)
dirac_delta[0] = 1

impulse_response = movingAvg(dirac_delta, 1, 9, 8, 7, 0)
print(f"Test for #2.), b1=1, b2=9, b3=8, xm1=7, xm2=0 : {movingAvg(dirac_delta, 1, 9, 8, 7, 0)}")

# plt.figure(figsize=(8,5)) # set size of plot
# plt.stem(impulse_response, label='Impulse Response', basefmt=" ")
# plt.xlabel(f'N=0,...,{N_2 - 1}')
# plt.ylabel('Impulse y[n]=h[n] * dirac[n]')
# plt.title('Impulse Response of movingAvg: b1=1, b2=9, b3=8, xm1=7, xm2=0')
# plt.grid(True)
# plt.show()

# The impulse response coefficients are directly related to the function inputs I picked, as
# y[0] = b1 * x[0] + b2 * xm1 + b3 * xm2 = 1 * 1 + 9 * 7 + 8 * 0 = 1 + 63 = 64
# y[1] = b1 * x[1] + b2 * x[0] + b3 * xm1 = 0 + 9 * 1 + 8 * 7 = 9 + 56 = 65
# y[3] = b1 * 0 + b2 * 0 + b3 * x[0] = 8 * 1 = 8

# 3. Create vector x, repeating five 1s, then five 0s, two times so total vector length = 20
# Calculate y[n] as y1, from moving average system for vector x, where b1=b2=b3=1/3, xm1=xm2=0 (rest)
pattern = [1] * 5 + [0] * 5
x_fives = np.tile(pattern, 2)
print(f"x_fives: {x_fives}")

# Plot the input x[n] and output y1 on the same plot
y_1 = movingAvg(x_fives, 1/3, 1/3, 1/3, 0, 0)
# plt.figure(figsize=(8,5)) # set size of plot
# plt.stem(y_1, markerfmt='bD', label='y1[n]', basefmt=" ")
# plt.stem(x_fives, markerfmt='r', label='x[n]', basefmt=" ")
# plt.legend()
# plt.xlabel(f'N=0,...,19')
# plt.ylabel('Moving Average, y_1[n]')
# plt.title('movingAvg of x[n]: b1=b2=b3=1/3, xm1=xm2=0')
# plt.grid(True)
# plt.show()

# Add an appropriate legend to label the two signals
# Is the input modified in a way that matches your intuition?
# Describe how the output relates to the input.
# Yes, the input is modified in a way that matches my intuition, the changes in the output are delayed by 3 samples,
# and as there are five 1's, the average maximum will still be 1.

# Calculate y2 on x, now b1=0.15, b2=0.7, b3=0.15, same xm's.
# Plot x[n] and y2 on same plot, with appropriate legend.
y_2 = movingAvg(x_fives, 0.15, 0.7, 0.15, 0, 0)
# plt.figure(figsize=(8,5)) # set size of plot
# plt.stem(y_2, markerfmt='bD', label='y2[n]', basefmt=" ")
# plt.stem(x_fives, markerfmt='r', label='x[n]', basefmt=" ")
# plt.legend()
# plt.xlabel(f'N=0,...,19')
# plt.ylabel('Moving Average, y_2[n]')
# plt.title('movingAvg of x[n]: b1=0.15, b2=0.7, b3=0.15, xm1=xm2=0')
# plt.grid(True)
# plt.show()

# How does this output look different than the previous plot? Why does it look different?
# This plot looks different, as the rate of change in the average is different. This is because the weights in
# our average are different, but the maximum is still the same as the weights still sum to 1.

# Calculate y3 on x, now b1=b2=b3=1/3, but xm1=7, xm2=10
y_3 = movingAvg(x_fives, 1/3, 1/3, 1/3, 7, 10)
# plt.figure(figsize=(8,5)) # set size of plot
# plt.stem(y_3, markerfmt='bD', label='y3[n]', basefmt=" ")
# plt.stem(x_fives, markerfmt='r', label='x[n]', basefmt=" ")
# plt.legend()
# plt.xlabel(f'N=0,...,19')
# plt.ylabel('Moving Average, y_3[n]')
# plt.title('movingAvg of x[n]: b1=b2=b3=1/3, xm1=7, xm2=10')
# plt.grid(True)
# plt.show()

# Plot y1 and y2 on the same plot, with appropriate legend.
# plt.figure(figsize=(8,5)) # set size of plot
# plt.stem(y_1, markerfmt='bD', label='y_1[n]', basefmt=" ")
# plt.stem(y_2, markerfmt='r', label='y_2[n]', basefmt=" ")
# plt.legend()
# plt.xlabel(f'N=0,...,19')
# plt.ylabel('Moving Averages')
# plt.title('movingAvg of x[n], as y_1[n] and y_2[n]')
# plt.grid(True)
# plt.show()

# Describe the differences you see in the two outputs.
# The difference in the outputs is that the rate of change is different in the two outputs.

# 4. The moving average filter is non-recursive, an FIR, so the convolution can be used.
# For b1=b2=b3=1/3, create vector h for h[n], impulse response
N_4 = 20
dirac_delta_h = np.zeros(N_4)
dirac_delta_h[0] = 1

impulse_response_h = movingAvg(dirac_delta, 1/3, 1/3, 1/3, 0, 0)

# Calculate the convolution of h[n] and x[n] from the last step (fives) and store as y4
y_4 = np.convolve(impulse_response_h, x_fives)
plt.stem(y_1, markerfmt='rD', label='y_1[n]', basefmt=" ")
plt.stem(y_3, markerfmt='g', label='y_3[n]', basefmt=" ")
plt.stem(y_4, markerfmt='bx', label='y_4[n]', basefmt=" ")
plt.legend()
plt.xlabel(f'N')
plt.ylabel('y_1[n], y_3[n], and y_4[n]')
plt.title('movingAvg y_1[n], y_3[n] of x[n], vs. Convolution of movingAvg and x[n]')
plt.grid(True)
plt.show()

# Compared to y1 and y3, are the number of outputs in y4 the same? How are the values related?
# What can you then say about the assumptions the convolution function makes about the initial system conditions?
# The number of outputs in y4 are not the same as y1 and y3, as it has N=28 instead of 20.
# The output values in the same range of N=20 are identical to y_1[n].
# I can then make the assumption that the convolution of the impulse response of the movingAvg convolved with x[n]
# is identical to applying movingAvg directly to x[n].

