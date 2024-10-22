# EE125 Project 2 - Part 1, Aliasing due to Undersampling
# Sept. 23, 2024

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Chirp signal given by (1): x(t) = sin(Omega_0 * t + 1/2 Beta * t^2)
# \Omega_0 is a frequency in rad/sec = 2piF, F is in Hz

# Instantaneous freq. of chirp (2) = derivative of the phase
#                                    (deriv. of argument of sin)
# \Theta_{inst}(t) = \frac{d}{dt} (\Omega_0 t + (1/2 * \Beta t^2)
#                  = \Omega_0 + \Beta t

# a.) \Omega_0 = 2\pi(1500) rad/sec and \Beta = 2\pi(3000) rad/sec^2
#     Calculate x(t) between 0 and 2 sec using chirp signal
#     pta = part a

T_pta = 2       # seconds
Fs_pta = 8192   # Hz

omegazero_pta = 2 * np.pi * 1500    # rad/sec
beta_pta = 2 * np.pi * 3000         # rad/sec^2

linspace_pta = np.linspace(0, T_pta, T_pta * Fs_pta)

x_pta = np.sin(omegazero_pta * linspace_pta + 0.5 * beta_pta * (linspace_pta ** 2))


# b.) Plot vector to determine approximate min and max frequency

# figure, (sp1, sp2, sp3, sp4, sp5) = plt.subplots(5, 1, figsize=(20,15))
# sp1.step(linspace_pta[ : x_pta.size//5],
#          x_pta[ : x_pta.size//5],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp1.set_ylabel('x(t)')
# sp1.set_title('Sinusoid Output of x(t), at F_s=8193Hz for t=2s')
# sp1.grid(True)
# sp2.step(linspace_pta[x_pta.size//5 : 2 * x_pta.size//5],
#          x_pta[x_pta.size//5 : 2 * x_pta.size//5],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp2.set_ylabel('x(t)')
# sp2.grid(True)
# sp3.step(linspace_pta[2 * x_pta.size//5 : 3 * x_pta.size//5],
#          x_pta[2 * x_pta.size//5 : 3 * x_pta.size//5],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp3.set_ylabel('x(t)')
# sp3.grid(True)
# sp4.step(linspace_pta[3 * x_pta.size//5 : 4 * x_pta.size//5],
#          x_pta[3 * x_pta.size//5 : 4 * x_pta.size//5],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp4.set_ylabel('x(t)')
# sp4.grid(True)
# sp5.step(linspace_pta[4 * x_pta.size//5 : ],
#          x_pta[4 * x_pta.size//5 : ],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp5.set_xlabel('t, in s')
# sp5.set_ylabel('x(t)')
# sp5.grid(True)
#
# plt.tight_layout()
# plt.show()

# sd.play(x_pta, Fs_pta)
# sd.wait()

# Comments:
#   max freq happens around .85 to .86, most dense
#   min freq happens around the end, 2s, from listening to it, also least dense.

# c.) Create longer 5 sec chirp, determine number of times signal reaches max/min
#     Determine approximate maxima and minima's

T_ptb = 5 # sec

linspace_ptb = np.linspace(0, T_ptb, T_ptb * Fs_pta)

x_ptb = np.sin(omegazero_pta * linspace_ptb + 0.5 * beta_pta * (linspace_ptb ** 2))
#
# figure, (sp1, sp2, sp3, sp4, sp5) = plt.subplots(5, 1, figsize=(20,15))
# sp1.step(linspace_ptb[ : x_ptb.size//5],
#          x_ptb[ : x_ptb.size//5],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp1.set_ylabel('x(t)')
# sp1.set_title('Sinusoid Output of x(t), at F_s=8193Hz for t=5s')
# sp1.grid(True)
# sp2.step(linspace_ptb[x_ptb.size//5 : 2 * x_ptb.size//5],
#          x_ptb[x_ptb.size//5 : 2 * x_ptb.size//5],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp2.set_ylabel('x(t)')
# sp2.grid(True)
# sp3.step(linspace_ptb[2 * x_ptb.size//5 : 3 * x_ptb.size//5],
#          x_ptb[2 * x_ptb.size//5 : 3 * x_ptb.size//5],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp3.set_ylabel('x(t)')
# sp3.grid(True)
# sp4.step(linspace_ptb[3 * x_ptb.size//5 : 4 * x_ptb.size//5],
#          x_ptb[3 * x_ptb.size//5 : 4 * x_ptb.size//5],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp4.set_ylabel('x(t)')
# sp4.grid(True)
# sp5.step(linspace_ptb[4 * x_ptb.size//5 : ],
#          x_ptb[4 * x_ptb.size//5 : ],
#          linestyle='-', lw='.1', color='green', label='sinusoid')
# sp5.set_xlabel('t, in s')
# sp5.set_ylabel('x(t)')
# sp5.grid(True)
#
# plt.tight_layout()
# plt.show()
#
sd.play(x_ptb, Fs_pta)
sd.wait()

# Comments:
#   Max freq: around .82 to .89 and 3.57 to 3.62
#   Min freq: around 2.2 and 4.96

# d.) Give short discussion on why signal changed!
#       The signal freq. kept increasing, but as we sample at Fs = 8192, our nyquist
#       freq. dictates that at Fs/2 (and multiples), the signal aliases to where
#       the frequency stops increasing and instead decreases. Then, as we approach
#       Fs, the same occurs where we increase up to the nyquist, then decrease after
#       reaching it.

# e.) Derive the formula to predict times at which sampled reaches max/min pitch
#       For instantaneous frequency, it constantly increases. So, to describe
#       the min and max behaviors, we must consider the multiples of Fs/2, aka Nyquist.
#       Fs = 8192 Hz = 8192 * 2pi rad/sec
#       Formula: omegazero + beta * t = n * Fs/2

# f.) See fig2 in graphs folder.
# Comments:
#   For 0.5s, there isn't enough time for the chirp to reach Fs/2. Therefore, no aliasing is present.
#   For 0.8s, the signal just about reaches Fs/2, no aliasing occurs and instead, we see the
#       that in the frequency domain, it has a greater region compared to 0.5Hz,
#       and that region reaches about 4000Hz, which is close to Fs/2.
#   For 1.2s, we observe aliasing, where around 3000Hz, we begin to see rapid oscillations
#       to the max frequency of the signal. This can be explained by the effects of aliasing
#       and the Nyquist. Once the chirp reaches Fs/2, it decreases (around 0.82 sec) in
#       frequency due to the aliasing, causing the amplitude at those frequencies to cover a
#       greater range. You could think of the frequency graph observing a "doubling" in the
#       range between Fs/2 and the frequencies below it. But, as 1.2s is too short for
#       the chirp to reach Fs, that decrease stops around 3000Hz.
#   For 2s, now we observe aliasing over a greater region of the frequency domain. Referring to
#       the previous explanation, 2s allows for the signal to reach two multiples of
#       Fs/2. Then, as the amplitude increased and decreased for multiple times over those
#       frequency ranges, they have this rapidly oscillating appearance.
