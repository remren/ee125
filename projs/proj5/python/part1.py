# EE125 Project 5 - Part 1, Using DFT to Estimate Amplitude of a Sinusoid
# Nov. 29, 2024

import numpy as np
import matplotlib.pyplot as plt

# 1.) DTFT of a truncated sinusoid
M = 256
# M = 512
n = np.arange(M)
w0 = 0.2531 * np.pi
A = 1
x = A * np.cos(w0*n)

# 2.) DFT of the truncated sinusoid
N = 1024
X = np.fft.fft(x, N)
k = np.arange(N)
k1 = int(N * w0 / (2 * np.pi))
k0 = int(N - k1)
peaks = [np.abs(X[k1]), np.abs(X[k0])]
# plt.stem(k, np.abs(X), markerfmt='.')
# plt.annotate(f'k0: {peaks[0]:.2f}',
#              xy=(k[128], peaks[0]),
#              xytext=(k[128] + 10, peaks[0] + 0.25),
#              arrowprops=dict(facecolor='r', headwidth=4, shrink=0.05))
# plt.annotate(f'k1: {peaks[1]:.2f}',
#              xy=(k[896], peaks[0]),
#              xytext=(k[896] + 10, peaks[0] + 0.25),
#              arrowprops=dict(facecolor='r', headwidth=4, shrink=0.05))
# plt.xlabel('DFT Sample k')
# plt.ylabel('| X(k) |')
# plt.title(f'DFT Magnitude of truncated sinusoid')
# plt.grid()
# plt.show()

# 3.) Fixing the Amplitude
# 4.) Test the estimation function
# 5.) Test with other inputs
def plotSpectrumForAmplitudeEstimation(x, N):
    M = len(x)
    X = np.fft.fft(x, N)
    kpos = np.arange(1, int(np.floor(N / 2 + 2)), dtype=int)  # Ensure integer indices
    Xhat = np.abs(X[kpos - 1]) / M  # Adjust index for Python (0-based)
    k2 = np.arange(2, int(np.ceil(N / 2 + 1)), dtype=int)  # Ensure integer indices
    Xhat[k2 - 1] = 2 * Xhat[k2 - 1]  # Adjust index for Python
    plt.stem(kpos, Xhat, markerfmt='.')
    plt.xlabel('DFT Sample k')
    plt.ylabel('|X(k)|')
    plt.grid(True)
    k0 = Xhat.argmax()
    plt.title(f'Case 3, DFT of truncated Sinusoid (A=1, M=256, N=1024, w=0.2531pi)')
    plt.annotate(f'k0: {Xhat[k0]:.2f}',
                 xy=(k0, Xhat[k0]),
                 xytext=(k0 + 25, Xhat[k0] - 0.05),
                 arrowprops=dict(facecolor='r', headwidth=4, shrink=0.05))
    plt.show()


plotSpectrumForAmplitudeEstimation(x, 1024)

# def plotWindowedSpectrumForAmpEstimationVer1(x, win, N, Fs):
