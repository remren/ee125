% EE125 Project 6 - FIR Filter Design
% December 7, 2024
%% Part 2: Equirriple Filter Design by Parks-McClellan
% Design a filter with this spec:
% Fs = 1000Hz, Fc = 100Hz (Lowpass), Ripple is max +/- 2%
% F > 175Hz, attenuation should be at least 40dB.

Fs = 1000;  % Hz
Fc = 100;   % Hz
Fstopband = 175; % Hz
max_passband_distortion = 0.02;
transitionband_width = 175 - Fc; % Hz
stopband_attenuation = 40; % dB

%% Determine Filter Length
% Given equation for estimation (too low, attenuation too low)
L_est = (stopband_attenuation * Fs) / (22 * transitionband_width);

% Equation 10.2.97? Not necessary, just increment our picked L.
% Needs to be odd, whole for low pass (one point at F = 0Hz)
% L = 25 doesn't work. Stopband Attenuation is not 40dB.
L_pick = 27;
order = L_pick - 1;

% firpm uses normalized frequency points. Need to normalize Fstopband, Fc.
Fc_norm = Fc * (2 / Fs);
Fstopband_norm = Fstopband * (2 / Fs);

% From range [0, 1], where 1 is the Nyquist Frequency (so why we normalize)
freq_pts = [0, Fc_norm, Fstopband_norm, 1];

% Desired amplitudes at frequencies between pairs of points
% So from F = 0 to F = Fc_norm, desired amplitude is 1 for LPF.
desired_amp = [1, 1, 0, 0];

%% Run the filter designer :D
h = firpm(order, freq_pts, desired_amp);

H = freqz(h);

% Plot with the normalized frequencies
f_norm = linspace(0, Fs/2, length(H));

% Error
plot(f_norm, abs(H));
% For dB
% plot(f_norm, mag2db(abs(H)));
xlabel("Frequency (Hz)");
% ylabel("Magnitude (dB)");
ylabel("Magnitude (Linear)");
title("Frequency Magnitude Response for Equiripple Filter, L=27");
grid on
