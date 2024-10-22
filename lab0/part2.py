# EE125 Project 1 - Part 2, Random Numbers and Plotting
# Sept. 16, 2024

import numpy as np
import matplotlib.pyplot as plt

# 1. Set the seed of the RNG to 12345
seed = 12345
rng = np.random.default_rng(seed)

# 2. Create vector x, containing N=10,000 samples
#    drawing from a Normal prob. dist. w/mean u=0 variance=1
N = 10000
mean = 0
variance = 1
x = rng.normal(loc=mean, scale=variance, size=N)

# 3. Plot the vector x, with the first 10 samples associated with sample index k=0
k = np.arange(10) # k=0,..,9
x_sampled = x[:10] # first 10 samples

plt.figure(figsize=(8,5)) # set size of plot
plt.plot(k, x_sampled, marker='o', linestyle='-', color='green', label='First 10 Samples')

#    Label x and y axis w/text, adding title "Ten Samples from a Normal Distribution"
plt.xlabel('Sample Index (k, for k=0,...,9)')
plt.ylabel('Values from Vector x(k)')
plt.title('Ten Samples from a Normal Distribution')

#    Adjust y axis to range [-R, R], wh. R=1.2*max{|x(k)|} for k=0,...,9
R = 1.2 * max(abs(x_sampled))
plt.ylim(-R, R)

#    Add a grid to the plot.
plt.grid(True)

#    Add a legend in default location w/string "X ~ N(0,1)" to describe entries of x are on a norm. dist.
plt.legend("X ~ N(0,1)")

# Show the plot
plt.show()

# 4. What did you learn from this plotting exercise?
#    I primarily learned how to utilize numpy and matplotlib to produce nice looking graphs in Python.
#    More importantly, also how to fill these graphs with vital information to inform the viewer the
#    details about the contents of the PMF and PDF.
