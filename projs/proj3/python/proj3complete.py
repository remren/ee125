# EE125 Project 3 - FIR and IIR Filtering of Nerve Signals
# October 21, 2024

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pymatreader as pmr

data = pmr.read_mat('NerveData_Project3.mat') # Load data from .mat file
print(f"types {[k for k in data]}")           # Find the keys in dictionary

fs_hZ                   = int(data["fs_hZ"])
cleanSnap_uV            = np.array(data["cleanSnap_uV"])
contamSnapPowerline_uV  = np.array(data["contamSnapPowerline_uV"])
contamSnapBaseline_uV   = np.array(data["contamSnapBaseline_uV"])

# 1.) Plot the data
# All data is plotted at Fs = 10,000 Hz
# 12 ms data total, uV for data, time vector is in ms
t1 = 12         # in ms
Fs = int(fs_hZ) # in Hz
t1_ls = np.linspace(0, t1, t1 * Fs)
t1_space = np.linspace(0, 12, 120)
# plt.plot(cleanSnap_uV)
# plt.show()

# 2.) Create the analysis function
def analyzeSNAP(x, Fs):
    """
        Analyzes SNAP signal and returns several helpful parameters
        :param x:   SNAP Signal x(n), to analyze. Should be a numpy array.
        :param Fs:  Sampling frequency as an integer in Hz
        :return:    ampl, the peak to peak amplitude
                    latency, the time it takes in s, for the signal to reach its max
                    amax, the index in x where the max amplitude occurs
                    amin, the index in x where the min amplitude occurs
                    tmin, the time it takes in s, for the signal to reach its min
        """
    amax = max(x)
    amin = min(x)
    ampl = amax - amin
    latency = x.argmax() / fs_hZ
    tmin = x.argmin() / fs_hZ
    return [ampl, latency, amax, amin, tmin]

# test = analyzeSNAP(cleanSnap_uV, fs_hZ)
# print(test)
# print(f"cleanSnap max index: {np.argmax(cleanSnap_uV)}, value: {cleanSnap_uV[np.argmax(cleanSnap_uV)]}")

# 3.) Analyze reference SNAP
cleansnap_analyze = analyzeSNAP(cleanSnap_uV, fs_hZ)
# print(f"{cleansnap_analyze}")
# ampl, latency, amax, amin, and tmin
# [np.float64(39.47167015363368), np.float64(0.0035), np.float64(20.835104751011546),
#  np.float64(-18.636565402622136), np.float64(0.0044)]
# plt.plot(cleanSnap_uV, marker='.', label='cleanSnap')
# plt.xlabel('t, in milliseconds')
# plt.ylabel(r'Signal Output in uV')
# cs_max_index = cleanSnap_uV.argmax()
# cs_min_index = cleanSnap_uV.argmin()
# # Annotate the latency (as well as amax)
# plt.annotate(f'latency: {(cleansnap_analyze[1]*fs_hZ):.2f} ms, amax: {cleansnap_analyze[2]:.2f} uV',
#              xy=(cs_max_index, cleanSnap_uV[cs_max_index]),
#              xytext=(cs_max_index + 6, cleanSnap_uV[cs_max_index] - 0.25),
#              arrowprops=dict(facecolor='r', headwidth=4, shrink=0.05),
#              color='r')
# plt.annotate(f'tmin: {(cleansnap_analyze[4]*fs_hZ):.2f} ms, amin: {cleansnap_analyze[3]:.2f} uV',
#              xy=(cs_min_index, cleanSnap_uV[cs_min_index]),
#              xytext=(cs_min_index + 6, cleanSnap_uV[cs_min_index] - 0.25),
#              arrowprops=dict(facecolor='r', headwidth=4, shrink=0.05),
#              color='r')
# plt.title(f'cleanSnap Analysis, ampl_pp={cleansnap_analyze[0]:.2f}')
# plt.grid()
# plt.show()

# 4.) Design a IIR highpass and plot freq response and group delay in Hz
#       Given: 6th order Butterworth highpass filter, w/cutoff frequency of 100Hz
[b_iir, a_iir] = scipy.signal.butter(6,100/(Fs/2),'highpass')
# Computes frequency response
freq_iir, h_iir = scipy.signal.freqz(b_iir, a_iir) # default 512 points, or worN
# Convert freq. to Hz
freq_hz_iir = freq_iir * fs_hZ / (2 * np.pi)
# Get magnitude in decibels
mag_db_iir = 20 * np.log10(np.abs(h_iir[1:])) # took out first value, as it is zero.

# Get group delay
freq_grpdelay_iir, grpdelay_iir = scipy.signal.group_delay([b_iir, a_iir], fs=fs_hZ)
# Convert freq. to Hz??? Confused. I don't think this is necessary...
# freq_grpdelay_hz_iir = freq_grpdelay_iir * fs_hZ / (2 * np.pi)

# Plotting time!
# plt.subplot(2, 1, 1)
# plt.plot(freq_hz_iir[1:], mag_db_iir) # took out first value in mag_dec, likewise in freq_hz
# plt.annotate(f'60 Hz',
#              xy=(60 , mag_db_iir[4] + 12),
#              xytext=(60 + 20, mag_db_iir[4]),
#              arrowprops=dict(facecolor='r', headwidth=4, shrink=0.05),
#              color='r')
# plt.title('Mag. Response of 6th-order Butterworth Highpass, 20log_10|H(F)|')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.grid(True)
# # plt.xlim(0, 500)  # Zoom in to appropriate frequency range
# plt.ylim(-125, 5)  # Adjust magnitude range for clarity
#
# # Plot the group delay
# plt.subplot(2, 1, 2)
# plt.plot(freq_grpdelay_iir, grpdelay_iir)
# plt.title('Group Delay')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Group Delay (samples)')
# plt.grid(True)
# # plt.xlim(0, 500) # Zoom in to the same range for consistency
#
# # Show the plots
# plt.tight_layout()
# plt.show()

# 5.) Design a FIR highpass

# produces an array with (length), eg. 52 coefficients. NOT FILTER ORDER.
# Filter order is coefficients - 1. So 52 coefficients -> 51st-order
# cannot work with an even length, as an even length = even # of coefficients
# which must have a zero response at Nyquist freq. Thus, we give length=53
b_fir = scipy.signal.firwin(53, 100/(fs_hZ/2), pass_zero='highpass')

# Plot Mag and GRPDELAY

# Get frequency response of FIR
freq_fir, h_fir = scipy.signal.freqz(b_fir) # default 512 points, or worN
# Convert freq to Hz
freq_hz_fir = freq_fir * fs_hZ / (2 * np.pi)
# Get magnitude in dB
mag_db_fir = 20 * np.log10(np.abs(h_fir))

# Get group delay
freq_grpdelay_fir, grpdelay_fir = scipy.signal.group_delay([b_fir, 1], fs=fs_hZ)

# Plotting time!
plt.subplot(2, 1, 1)
plt.plot(freq_hz_fir, mag_db_fir)
plt.annotate(f'60 Hz',
             xy=(60 , mag_db_fir[4] + 12),
             xytext=(60 + 20, mag_db_fir[4]),
             arrowprops=dict(facecolor='r', headwidth=4, shrink=0.05),
             color='r')
plt.title('Mag. Response of 52nd-order FIR Highpass at 20log_10|H(F)|')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.xlim(0, 800)  # zoom freq
plt.ylim(-20, 5)  # zoom mag

# Plot the group delay
plt.subplot(2, 1, 2)
plt.plot(freq_grpdelay_fir, grpdelay_fir)
plt.title('Group Delay')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Group Delay (samples)')
plt.grid(True)
plt.xlim(0, 800) # zoom freq
plt.ylim(20, 30)  # zoom mag

# Show the plots
plt.tight_layout()
plt.show()

# Discussion
# Discussion
# 6.) Explore computational efficiency
# Discussion
# 7.) Filter the reference signal
# 7a.) Discussion
# 7b.) Discussion
# 8.) Filter the contaminated signals
# 8a.) 2x2 subplot
# 8b.) Discussion
# 9.) Investigate Baseline Removal technique
# Discussion
# 10.) Remove phase distortion
# Discussion