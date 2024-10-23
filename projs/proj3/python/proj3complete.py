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
                    amax, the max amplitude in the signal
                    amin, the max amplitude in the signal
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

# # print(f"{cleansnap_analyze}")
# # ampl, latency, amax, amin, and tmin
# # [np.float64(39.47167015363368), np.float64(0.0035), np.float64(20.835104751011546),
# #  np.float64(-18.636565402622136), np.float64(0.0044)]
# plt.plot(t1_space, cleanSnap_uV, marker='.', label='cleanSnap')
# plt.xlabel('t, in milliseconds')
# plt.ylabel(r'Signal Output in uV')

cs_max_index = cleanSnap_uV.argmax()
cs_min_index = cleanSnap_uV.argmin()

# # Annotate the latency (as well as amax)
# plt.annotate(f'latency: {(cleansnap_analyze[1]*fs_hZ / 10):.2f} ms, amax: {cleansnap_analyze[2]:.2f} uV',
#              xy=(cs_max_index / 10, cleanSnap_uV[cs_max_index]),
#              xytext=(cs_max_index / 10 + 1, cleanSnap_uV[cs_max_index] - 0.25),
#              arrowprops=dict(facecolor='b', headwidth=4, shrink=0.05),
#              color='b')
# plt.annotate(f'tmin: {(cleansnap_analyze[4]*fs_hZ / 10):.2f} ms, amin: {cleansnap_analyze[3]:.2f} uV',
#              xy=(cs_min_index / 10, cleanSnap_uV[cs_min_index]),
#              xytext=(cs_min_index / 10 + 1, cleanSnap_uV[cs_min_index] - 0.25),
#              arrowprops=dict(facecolor='b', headwidth=4, shrink=0.05),
#              color='b')
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

# # Plotting time!
# plt.subplot(2, 1, 1)
# plt.plot(freq_hz_fir, mag_db_fir)
# plt.annotate(f'60 Hz',
#              xy=(60 , -5.7),
#              xytext=(60 + 30, -10),
#              arrowprops=dict(facecolor='r', headwidth=4, shrink=0.05),
#              color='r')
# plt.title('Mag. Response of 52nd-order FIR Highpass at 20log_10|H(F)|')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.grid(True)
# plt.xlim(0, 800)  # zoom freq
# plt.ylim(-20, 5)  # zoom mag
#
# # Plot the group delay
# plt.subplot(2, 1, 2)
# plt.plot(freq_grpdelay_fir, grpdelay_fir)
# plt.title('Group Delay')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Group Delay (samples)')
# plt.grid(True)
# plt.xlim(0, 800) # zoom freq
# plt.ylim(20, 30)  # zoom mag
#
# # Show the plots
# plt.tight_layout()
# plt.show()

# Discussion: its about -5.82 dB. same reasoning as last part
# Discussion: There will be magnitude loss in the 200~350 ish frequency band
#             There will not be distortion, but the signal is at a constant delay.

# 6.) Explore computational efficiency
# Discussion: If efficiency metric is defined as the number of multiplies per output
#             sample, the filter that is computationally more efficient is
#             both, assuming done via FFt, for the same number of points
#             FIR Window        - 52nd order, so 52 multiplies
#             IIR Butterworth   - 6th order, so 6 multiplies

# 7.) Filter the reference signal
# 6th-order Butterworth
cleanSnap_uV_iir_filtered = scipy.signal.lfilter(b_iir, a_iir, cleanSnap_uV)
# 52nd-order FIR Window
cleanSnap_uV_fir_filtered = scipy.signal.lfilter(b_fir, 1, cleanSnap_uV)

# # Plotting time!
# plt.plot(t1_space, cleanSnap_uV)
cleansnap_analyze = analyzeSNAP(cleanSnap_uV, fs_hZ)
cleanSnap_max_index = cleanSnap_uV.argmax()
# print(cleanSnap_max_index)
# # annotate ampl and latency for clean signal
# plt.annotate(f'ampl (pk to pk): {cleansnap_analyze[0]:.2f} uV, '
#              f'latency: {(cleansnap_analyze[1]*fs_hZ / 10):.2f} ms',
#              xy=(cs_max_index / 10, cleanSnap_uV[cs_max_index]),
#              xytext=(cs_max_index / 10 + 1, cleanSnap_uV[cs_max_index] - 0.25),
#              arrowprops=dict(facecolor='b', headwidth=4, shrink=0.05),
#              color='b')
#
# plt.plot(t1_space, cleanSnap_uV_fir_filtered)
# # analyze fir filtered
firfiltered_analyze = analyzeSNAP(cleanSnap_uV_fir_filtered, fs_hZ)
csfirfilt_maxindex = cleanSnap_uV_fir_filtered.argmax()
# print(csfirfilt_maxindex)
# # annotate ampl and latency for fir filtered
# plt.annotate(f'ampl: {firfiltered_analyze[0]:.2f} uV, '
#              f'latency: {(firfiltered_analyze[1]*fs_hZ / 10):.2f} ms',
#              xy=(firfiltered_analyze[1] * fs_hZ / 10, firfiltered_analyze[2]),
#              xytext=(firfiltered_analyze[1] * fs_hZ / 10 + 1, firfiltered_analyze[2] - 0.9),
#              arrowprops=dict(facecolor='darkorange', headwidth=2, shrink=0.03),
#              color='darkorange')
#
# plt.plot(t1_space, cleanSnap_uV_iir_filtered)
# # analyze iir filtered
iirfiltered_analyze = analyzeSNAP(cleanSnap_uV_iir_filtered, fs_hZ)
csiirfilt_maxindex = cleanSnap_uV_iir_filtered.argmax()
# print(csiirfilt_maxindex)
# # annotate ampl and latency for fir filtered
# plt.annotate(f'ampl: {iirfiltered_analyze[0]:.2f} uV, '
#              f'latency: {(iirfiltered_analyze[1]*fs_hZ / 10):.2f} ms',
#              xy=(iirfiltered_analyze[1] * fs_hZ / 10, iirfiltered_analyze[2]),
#              xytext=(iirfiltered_analyze[1] * fs_hZ / 10 + 1, iirfiltered_analyze[2] - 0.9),
#              arrowprops=dict(facecolor='green', headwidth=2, shrink=0.03),
#              color='green')
#
# plt.ylabel("Signal Output in uV")
# plt.xlabel("t, in ms")
# plt.title("SNAP Signal and Filtered Comparison")
# plt.grid()
# plt.legend(['cleanSnap', 'cleanSnap FIR Filtered', 'cleanSnap IIR Filtered'])
#
# plt.tight_layout()
# plt.show()

# 7a.) Discussion
#       FIR: Seems to have small changes to magnitude, such as a smaller ampl. However, the second maxima after
#            amax is greater. However, there is no distortion, just a constant delat in the signal due to the
#            group delay being constant. Upon closer inspection, the FIR mag response has a small gain at
#            a certain frequency range, which matches this behavior.
#       IIR: Due to the group delay having a non-linear relationship with the frequency, the signal is significantly
#            distorted. However, we can observe the magnitude response of the Butterworth filter, where the
#            the initial signal and signal after the spike/latency match the unfiltered output.
# 7b.) Discussion
#       Obeserved latency caused by FIR filter is 6.1 - 3.5ms = 2.6 ms
#       The shift for an FIR filter of length 21 (order 20), is probably 20 / 2, as the length 53, 52 order
#       has constant grpdelay of about 26
#       For a medical doctor, if they are only interested in measuring SNAP latency and not effects of signal
#       processing, then perhaps it would be wise to just plot the clean SNAP at the shifts as to remove the
#       additional info about signal processing.

# 8.) Filter the contaminated signals
#       IIR Baseline | IIR Powerline
#       FIR Baseline | IIR Powerline

# baseline
baseline_analyze = analyzeSNAP(contamSnapBaseline_uV, fs_hZ)
baseline_fir_filtered = scipy.signal.lfilter(b_fir,
                                             1,
                                             contamSnapBaseline_uV)
baseline_fir_analyze = analyzeSNAP(baseline_fir_filtered, fs_hZ)

baseline_iir_filtered = scipy.signal.lfilter(b_iir,
                                             a_iir,
                                             contamSnapBaseline_uV)
baseline_iir_analyze = analyzeSNAP(baseline_iir_filtered, fs_hZ)

# powerline
powerline_analyze = analyzeSNAP(contamSnapPowerline_uV, fs_hZ)
powerline_fir_filtered = scipy.signal.lfilter(b_fir,
                                              1,
                                              contamSnapPowerline_uV)
powerline_fir_analyze = analyzeSNAP(powerline_fir_filtered, fs_hZ)

powerline_iir_filtered = scipy.signal.lfilter(b_iir,
                                              a_iir,
                                              contamSnapPowerline_uV)
powerline_iir_analyze = analyzeSNAP(powerline_iir_filtered, fs_hZ)

# # 8a.) 2x2 subplot (contains contaminated input standalone
# #                   and filtered output)
# figure, axs = plt.subplots(2, 2, figsize=(18, 12))
#
# # print(contamSnapBaseline_uV.argmax())
# # print(contamSnapBaseline_uV[119])
# # print(contamSnapBaseline_uV)
#
# # iir baseline
# axs[0, 0].plot(t1_space, contamSnapBaseline_uV, marker='.')
# axs[0, 0].annotate(f'ampl: {baseline_analyze[0]:.2f} uV,\n'
#                    f'latency: {(baseline_analyze[1] *fs_hZ / 10 + 0.1):.2f}'
#                    f' ms',
#                    xy=(baseline_analyze[1] * fs_hZ / 10 + 0.1,
#                        baseline_analyze[2]),
#                    xytext=(baseline_analyze[1] * fs_hZ / 10 - 1.5,
#                            baseline_analyze[2] + 4),
#                    arrowprops=dict(facecolor='blue',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='blue')
# axs[0, 0].plot(t1_space, baseline_iir_filtered, marker='.')
# axs[0, 0].annotate(f'ampl (pk to pk): {baseline_iir_analyze[0]:.2f} uV,\n'
#                    f'latency: {(baseline_iir_analyze[1]*fs_hZ / 10):.2f} ms',
#                    xy=(baseline_iir_analyze[1] * fs_hZ / 10,
#                        baseline_iir_analyze[2]),
#                    xytext=(baseline_iir_analyze[1] * fs_hZ / 10 + 1,
#                            baseline_iir_analyze[2] - 0.9),
#                    arrowprops=dict(facecolor='red',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='red')
# axs[0, 0].set_ylabel("Signal Output in uV")
# axs[0, 0].set_xlabel("t, in ms")
# axs[0, 0].set_title("IIR of Baseline Interference")
# axs[0, 0].grid(True)
# axs[0, 0].legend(['Baseline Contaminated', 'Baseline IIR Filtered'])
#
# # iir powerline
# axs[0, 1].plot(t1_space, contamSnapPowerline_uV, marker='.')
# axs[0, 1].annotate(f'ampl: {powerline_analyze[0]:.2f} uV,\n'
#                    f'latency: {(powerline_analyze[1] *fs_hZ / 10):.2f} ms',
#                    xy=(powerline_analyze[1] * fs_hZ / 10,
#                        powerline_analyze[2]),
#                    xytext=(powerline_analyze[1] * fs_hZ / 10 + 1,
#                            powerline_analyze[2] + 3),
#                    arrowprops=dict(facecolor='blue',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='blue')
# axs[0, 1].plot(t1_space, powerline_iir_filtered, marker='.')
# axs[0, 1].annotate(f'ampl: {powerline_iir_analyze[0]:.2f} uV,\n'
#                    f'latency: {(powerline_iir_analyze[1]*fs_hZ / 10):.2f} ms',
#                    xy=(powerline_iir_analyze[1] * fs_hZ / 10,
#                        powerline_iir_analyze[2]),
#                    xytext=(powerline_iir_analyze[1] * fs_hZ / 10 + 1,
#                            powerline_iir_analyze[2] + 1),
#                    arrowprops=dict(facecolor='red',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='red')
# axs[0, 1].set_ylabel("Signal Output in uV")
# axs[0, 1].set_xlabel("t, in ms")
# axs[0, 1].set_title("IIR of Power-line Interference")
# axs[0, 1].grid(True)
# axs[0, 1].legend(['Power-line Contaminated', 'Power-line IIR Filtered'])
#
# # FIR baseline
# axs[1, 0].plot(t1_space, contamSnapBaseline_uV, marker='.')
# axs[1, 0].annotate(f'ampl: {baseline_analyze[0]:.2f} uV,\n'
#                    f'latency: {(baseline_analyze[1] *fs_hZ / 10 + 0.1):.2f}'
#                    f' ms',
#                    xy=(baseline_analyze[1] * fs_hZ / 10 + 0.1,
#                        baseline_analyze[2]),
#                    xytext=(baseline_analyze[1] * fs_hZ / 10 - 1.5,
#                            baseline_analyze[2] + 4),
#                    arrowprops=dict(facecolor='blue',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='blue')
# axs[1, 0].plot(t1_space, baseline_fir_filtered, marker='.')
# axs[1, 0].annotate(f'ampl: {baseline_fir_analyze[0]:.2f} uV,\n'
#                    f'latency: {(baseline_fir_analyze[1]*fs_hZ / 10):.2f} ms',
#                    xy=(baseline_fir_analyze[1] * fs_hZ / 10,
#                        baseline_fir_analyze[2]),
#                    xytext=(baseline_fir_analyze[1] * fs_hZ / 10 + 0.5,
#                            baseline_fir_analyze[2] - 1),
#                    arrowprops=dict(facecolor='red',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='red')
# axs[1, 0].set_ylabel("Signal Output in uV")
# axs[1, 0].set_xlabel("t, in ms")
# axs[1, 0].set_title("FIR of Baseline Interference")
# axs[1, 0].grid(True)
# axs[1, 0].legend(['Baseline Contaminated', 'Baseline FIR Filtered'])
#
# # FIR powerline
# axs[1, 1].plot(t1_space, contamSnapPowerline_uV, marker='.')
# axs[1, 1].annotate(f'ampl: {powerline_analyze[0]:.2f} uV,\n'
#                    f'latency: {(powerline_analyze[1] *fs_hZ / 10):.2f} ms',
#                    xy=(powerline_analyze[1] * fs_hZ / 10,
#                        powerline_analyze[2]),
#                    xytext=(powerline_analyze[1] * fs_hZ / 10 + 1,
#                            powerline_analyze[2] + 3),
#                    arrowprops=dict(facecolor='blue',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='blue')
# axs[1, 1].plot(t1_space, powerline_fir_filtered, marker='.')
# axs[1, 1].annotate(f'ampl: {powerline_fir_analyze[0]:.2f} uV,\n'
#                    f'latency: {(powerline_fir_analyze[1]*fs_hZ / 10):.2f} ms',
#                    xy=(powerline_fir_analyze[1] * fs_hZ / 10,
#                        powerline_fir_analyze[2]),
#                    xytext=(powerline_fir_analyze[1] * fs_hZ / 10 + 1,
#                            powerline_fir_analyze[2] + 1),
#                    arrowprops=dict(facecolor='red',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='red')
# axs[1, 1].set_ylabel("Signal Output in uV")
# axs[1, 1].set_xlabel("t, in ms")
# axs[1, 1].set_title("FIR of Power-line Interference")
# axs[1, 1].grid(True)
# axs[1, 1].legend(['Power-line Contaminated', 'Power-line FIR Filtered'])
#
# plt.tight_layout()
# plt.show()

# 8b.) Discussion
#       In baseline, needed to make a manual adjustment to latency, where due to 0-based indexing
#       and the total signal length being 0 to 12 ms, the last, and maxima point is at 12ms, but as ts
#       the largest at index 119, there is a mismatch.
#       Also, probably anything that has a latency of 0. Meaningfully, that doesn't make sense due to physics.
#       Cause??

# 9.) Investigate Baseline Removal technique x_m[n] = x[n] - x[1]
#       IIR Baseline | IIR Powerline
#       FIR Baseline | IIR Powerline

# removing baseline from contamSnapBaseline
baseline_baselineremoval = np.array([x - contamSnapBaseline_uV[0]
                                     for x in contamSnapBaseline_uV])
# removing baseline from contamSnapPowerline
powerline_baselineremoval = np.array([x - contamSnapPowerline_uV[0]
                                      for x in contamSnapPowerline_uV])

# baseline - blr = baseline removed
baseline_blr_analyze = analyzeSNAP(baseline_baselineremoval, fs_hZ)
baseline_blr_fir_filtered = scipy.signal.lfilter(b_fir,
                                                 1,
                                                 baseline_baselineremoval)
baseline_blr_fir_analyze = analyzeSNAP(baseline_blr_fir_filtered, fs_hZ)

baseline_blr_iir_filtered = scipy.signal.lfilter(b_iir,
                                                 a_iir,
                                                 baseline_baselineremoval)
baseline_blr_iir_analyze = analyzeSNAP(baseline_blr_iir_filtered, fs_hZ)

# powerline
powerline_blr_analyze = analyzeSNAP(powerline_baselineremoval, fs_hZ)
powerline_blr_fir_filtered = scipy.signal.lfilter(b_fir,
                                                  1,
                                                  powerline_baselineremoval)
powerline_blr_fir_analyze = analyzeSNAP(powerline_blr_fir_filtered, fs_hZ)

powerline_blr_iir_filtered = scipy.signal.lfilter(b_iir,
                                                  a_iir,
                                                  powerline_baselineremoval)
powerline_blr_iir_analyze = analyzeSNAP(powerline_blr_iir_filtered, fs_hZ)

# # 2x2 subplot (contains baseline removed input standalone and filtered output)
# figure, axs = plt.subplots(2, 2, figsize=(18, 12))
#
# # iir baseline
# axs[0, 0].plot(t1_space, baseline_baselineremoval, marker='.')
# axs[0, 0].annotate(f'ampl: {baseline_blr_analyze[0]:.2f} uV,\n'
#                    f'latency: {(baseline_blr_analyze[1] *fs_hZ / 10 + 0.1):.2f}'
#                    f' ms',
#                    xy=(baseline_blr_analyze[1] * fs_hZ / 10 + 0.1,
#                        baseline_blr_analyze[2]),
#                    xytext=(baseline_blr_analyze[1] * fs_hZ / 10 - 2.1,
#                            baseline_blr_analyze[2] + 1.3),
#                    arrowprops=dict(facecolor='blue',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='blue')
# axs[0, 0].plot(t1_space, baseline_blr_iir_filtered, marker='.')
# axs[0, 0].annotate(f'ampl (pk to pk): {baseline_blr_iir_analyze[0]:.2f} uV,\n'
#                    f'latency: {(baseline_blr_iir_analyze[1]*fs_hZ / 10):.2f}'
#                    f' ms',
#                    xy=(baseline_blr_iir_analyze[1] * fs_hZ / 10,
#                        baseline_blr_iir_analyze[2]),
#                    xytext=(baseline_blr_iir_analyze[1] * fs_hZ / 10 - 1.25,
#                            baseline_blr_iir_analyze[2] + 5),
#                    arrowprops=dict(facecolor='red',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='red')
# axs[0, 0].set_ylabel("Signal Output in uV")
# axs[0, 0].set_xlabel("t, in ms")
# axs[0, 0].set_title("IIR of Baseline Interference, with Baseline Removal")
# axs[0, 0].grid(True)
# axs[0, 0].legend(['Baseline Contaminated, with Baseline Removal',
#                   'IIR Filtered'])
#
# # iir powerline
# axs[0, 1].plot(t1_space, powerline_baselineremoval, marker='.')
# axs[0, 1].annotate(f'ampl: {powerline_blr_analyze[0]:.2f} uV,\n'
#                    f'latency: {(powerline_blr_analyze[1] *fs_hZ / 10):.2f}'
#                    f' ms',
#                    xy=(powerline_blr_analyze[1] * fs_hZ / 10,
#                        powerline_blr_analyze[2]),
#                    xytext=(powerline_blr_analyze[1] * fs_hZ / 10 + 0.5,
#                            powerline_blr_analyze[2] + 4.9),
#                    arrowprops=dict(facecolor='blue',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='blue')
# axs[0, 1].plot(t1_space, powerline_blr_iir_filtered, marker='.')
# axs[0, 1].annotate(f'ampl: {powerline_blr_iir_analyze[0]:.2f} uV,\n'
#                    f'latency: {(powerline_blr_iir_analyze[1]*fs_hZ / 10):.2f}'
#                    f' ms',
#                    xy=(powerline_blr_iir_analyze[1] * fs_hZ / 10,
#                        powerline_blr_iir_analyze[2]),
#                    xytext=(powerline_blr_iir_analyze[1] * fs_hZ / 10 - 0.75,
#                            powerline_blr_iir_analyze[2] - 30),
#                    arrowprops=dict(facecolor='red',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='red')
# axs[0, 1].set_ylabel("Signal Output in uV")
# axs[0, 1].set_xlabel("t, in ms")
# axs[0, 1].set_title("IIR of Power-line Interference, with Baseline Removal")
# axs[0, 1].grid(True)
# axs[0, 1].legend(['Power-line Contaminated, with Baseline Removal',
#                   'IIR Filtered'])
#
# # FIR baseline
# axs[1, 0].plot(t1_space, baseline_baselineremoval, marker='.')
# axs[1, 0].annotate(f'ampl: {baseline_blr_analyze[0]:.2f} uV,\n'
#                    f'latency: {(baseline_blr_analyze[1] *fs_hZ / 10 + 0.1):.2f}'
#                    f' ms',
#                    xy=(baseline_blr_analyze[1] * fs_hZ / 10 + 0.1,
#                        baseline_blr_analyze[2]),
#                    xytext=(baseline_blr_analyze[1] * fs_hZ / 10 - 2.1,
#                            baseline_blr_analyze[2] + 1.3),
#                    arrowprops=dict(facecolor='blue',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='blue')
# axs[1, 0].plot(t1_space, baseline_blr_fir_filtered, marker='.')
# axs[1, 0].annotate(f'ampl: {baseline_blr_fir_analyze[0]:.2f} uV,\n'
#                    f'latency: {(baseline_blr_fir_analyze[1]*fs_hZ / 10):.2f}'
#                    f' ms',
#                    xy=(baseline_blr_fir_analyze[1] * fs_hZ / 10,
#                        baseline_blr_fir_analyze[2]),
#                    xytext=(baseline_blr_fir_analyze[1] * fs_hZ / 10 - 0.8,
#                            baseline_blr_fir_analyze[2] + 5),
#                    arrowprops=dict(facecolor='red',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='red')
# axs[1, 0].set_ylabel("Signal Output in uV")
# axs[1, 0].set_xlabel("t, in ms")
# axs[1, 0].set_title("FIR of Baseline Interference, with Baseline Removal")
# axs[1, 0].grid(True)
# axs[1, 0].legend(['Baseline Contaminated, with Baseline Removal',
#                   'FIR Filtered'])
#
# # FIR powerline
# axs[1, 1].plot(t1_space, powerline_baselineremoval, marker='.')
# axs[1, 1].annotate(f'ampl: {powerline_blr_analyze[0]:.2f} uV,\n'
#                    f'latency: {(powerline_blr_analyze[1] *fs_hZ / 10):.2f}'
#                    f' ms',
#                    xy=(powerline_blr_analyze[1] * fs_hZ / 10,
#                        powerline_blr_analyze[2]),
#                    xytext=(powerline_blr_analyze[1] * fs_hZ / 10 + 0.5,
#                            powerline_blr_analyze[2] + 4.9),
#                    arrowprops=dict(facecolor='blue',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='blue')
# axs[1, 1].plot(t1_space, powerline_blr_fir_filtered, marker='.')
# axs[1, 1].annotate(f'ampl: {powerline_blr_fir_analyze[0]:.2f} uV,\n'
#                    f'latency: {(powerline_blr_fir_analyze[1]*fs_hZ / 10):.2f}'
#                    f' ms',
#                    xy=(powerline_blr_fir_analyze[1] * fs_hZ / 10,
#                        powerline_blr_fir_analyze[2]),
#                    xytext=(powerline_blr_fir_analyze[1] * fs_hZ / 10 + 1,
#                            powerline_blr_fir_analyze[2] - 2),
#                    arrowprops=dict(facecolor='red',
#                                    headwidth=2,
#                                    shrink=0.05),
#                    color='red')
# axs[1, 1].set_ylabel("Signal Output in uV")
# axs[1, 1].set_xlabel("t, in ms")
# axs[1, 1].set_title("FIR of Power-line Interference, with Baseline Removal")
# axs[1, 1].grid(True)
# axs[1, 1].legend(['Power-line Contaminated, with Baseline Removal',
#                   'FIR Filtered'])
#
# plt.tight_layout()
# plt.show()

# Discussion bnleghj

# 10.) Remove phase distortion
# Perform filtfilt and analyzeSNAP to graph IIR filter's response
#   on all 3 signals
clean_iir_filtfilt = scipy.signal.filtfilt(b_iir,
                                           a_iir,
                                           cleanSnap_uV)
clean_ffanalyze = analyzeSNAP(clean_iir_filtfilt, fs_hZ)

baseline_iir_filtfilt = scipy.signal.filtfilt(b_iir,
                                              a_iir,
                                              contamSnapBaseline_uV)
baseline_ffanalyze = analyzeSNAP(baseline_iir_filtfilt, fs_hZ)

powerline_iir_filtfilt = scipy.signal.filtfilt(b_iir,
                                               a_iir,
                                               contamSnapPowerline_uV)
powerline_ffanalyze = analyzeSNAP(powerline_iir_filtfilt, fs_hZ)

# Plotting time!
plt.plot(t1_space, clean_iir_filtfilt)
# annotate ampl and latency for the cleanSnap that's been filtfilt w/IIR filter
plt.annotate(f'ampl: {clean_ffanalyze[0]:.2f} uV, '
             f'latency: {(clean_ffanalyze[1]*fs_hZ / 10):.2f} ms',
             xy=(clean_ffanalyze[1] * fs_hZ / 10,
                 clean_ffanalyze[2]),
             xytext=(clean_ffanalyze[1] * fs_hZ / 10 + 1,
                     clean_ffanalyze[2] + 0.5),
             arrowprops=dict(facecolor='blue', headwidth=2, shrink=0.03),
             color='blue')

plt.plot(t1_space, baseline_iir_filtfilt)
plt.annotate(f'ampl: {baseline_ffanalyze[0]:.2f} uV, '
             f'latency: {(baseline_ffanalyze[1]*fs_hZ / 10):.2f} ms',
             xy=(baseline_ffanalyze[1] * fs_hZ / 10,
                 baseline_ffanalyze[2]),
             xytext=(baseline_ffanalyze[1] * fs_hZ / 10 + 1,
                     baseline_ffanalyze[2] - 2),
             arrowprops=dict(facecolor='darkorange', headwidth=2, shrink=0.03),
             color='darkorange')

plt.plot(t1_space, powerline_iir_filtfilt)
plt.annotate(f'ampl: {powerline_ffanalyze[0]:.2f} uV, '
             f'latency: {(powerline_ffanalyze[1]*fs_hZ / 10):.2f} ms',
             xy=(powerline_ffanalyze[1] * fs_hZ / 10,
                 powerline_ffanalyze[2]),
             xytext=(powerline_ffanalyze[1] * fs_hZ / 10 + 1,
                     powerline_ffanalyze[2] + 0.7),
             arrowprops=dict(facecolor='green', headwidth=2, shrink=0.03),
             color='green')

plt.ylabel("Signal Output in uV")
plt.xlabel("t, in ms")
plt.title("SNAP Signal and IIR Filtered Comparison, with Phase Distortion Removed")
plt.grid()
plt.legend(['cleanSnap, IIR filtered and removed Phase Distortion',
            'Baseline Contamination, IIR filtered and removed Phase Distortion',
            'Powerline Contamination, IIR filtered and removed Phase Distortion'])

plt.tight_layout()
plt.show()

# Discussion