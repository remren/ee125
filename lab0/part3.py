# EE125 Project 1 - Part 3, Statistic Tools and More Plotting
# Sept. 16, 2024

import numpy as np
import matplotlib.pyplot as plt

# Raw data from part 2:
# 1. Set the seed of the RNG to 12345
seed = 12345
rng = np.random.default_rng(seed)

# 2. Create vector x, containing N=10,000 samples
#    drawing from a Normal prob. dist. w/mean u=0 variance=1
N = 10000
mean = 0
variance = 1
x = rng.normal(loc=mean, scale=variance, size=N)

# Part 3 Questions
# 1. Plot all raw data w/empirically-estimated PDF in two subplots in a new figure
# Create 2 subplots within new figure, in a 1 row and 2 column pattern
figure, (subplot1, subplot2) = plt.subplots(1, 2, figsize=(15, 10))

# Plot all N samples connected by a line
subplot1.plot(np.arange(N), x, linestyle='-', lw='.2', color='blue', label='x Raw Data')

# Label all elements of axis appropriately, and include title.
subplot1.set_xlabel('Sample Index, N=10,000')
subplot1.set_ylabel('Value of Vector x[N]')
subplot1.set_title('Raw Data of 10,000 Samples, Vector x')

# Adjust y-axis limits to range [-R, R], where R=6variance
R = 6*variance
plt.ylim(-R, R)

# Add grid to plot
subplot1.grid(True)

# In the second column, plot the PDF for the data.
# Create histogram, density=True to normalize data
hist_data, bin_edges = np.histogram(x, bins=250, density=True)
subplot2.step(bin_edges[:-1], hist_data, color='g', label='Normalized Histogram')

# # Label all elements of axis appropriately, and include title.
# subplot2.set_xlabel('Sample Index, N=10,000')
# subplot2.set_ylabel('Empirically-estimated Distribution Values')
# subplot2.set_title('Empirically-estimated PDF of 10,000 Samples from Vector x')
# plt.ylim(0, 1)
#
# # Add grid to plot
# subplot2.grid(True)
# plt.show()

# 2. Plot theoetical probability distribution p(x) ontop of subplot 2.
# p(x) = 1 / (variance * np.sqrt(2*pi)) * e^(-.5((x-mean)/variance)^2)
# here, mean = 0, variance = 1
# p(x) = (1 / np.sqrt(2*pi)) * np.exp(-0.5 * x ^2)
linspace_x = np.linspace(np.min(x), np.max(x), N)
p_x = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.pow(linspace_x, 2))

# Plot ontop
subplot2.plot(linspace_x, p_x, color='r', label='Theoretical Normal PDF')
# Label all elements of axis appropriately, and include title.
subplot2.set_xlabel('Values of x')
subplot2.set_ylabel('Density')
subplot2.set_title('Empirically-estimated and Theoretical Normal PDF')
plt.ylim(0, 1)

# Add grid to plot
subplot2.grid(True)
plt.tight_layout()
plt.show()
