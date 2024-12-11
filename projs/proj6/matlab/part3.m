% EE125 Project 6 - FIR Filter Design
% December 7, 2024
%% Part 3: Filter Design by Windowing
% Design a filter with this spec:
% Fs = 1000Hz, Fc = 100Hz (Lowpass), Ripple is max +/- 2%
% F > 175Hz, attenuation should be at least 40dB.

Fs = 1000;  % Hz
Fc = 100;   % Hz
Fstopband = 175; % Hz
max_passband_distortion = 0.02;
stopband_attenuation = 40; % dB

%% Filter Design with Hamming Window, then Rectangular Window
% Need to increase filter L until filter meets desired characteristics.

L = 1000;
order = L - 1;
Fc_norm = Fc * (2 / Fs);
f_norm = linspace(0, Fs/2, length(H));
h = fir1(order, Fc_norm, hamming(L));
% h = fir1(order, Fc_norm, rectwin(L));
H = freqz(h);

% Cutoff Frequency and Passband Distortion
% plot(f_norm, mag2db(abs(H)))
plot(f_norm, abs(H))
xlabel("Frequency (Hz)");
% ylabel("Magnitude (dB)");
ylabel("Magnitude (Linear)");
% title(['Magnitude Response for Hamming, L=' num2str(L)]);
title(['Magnitude Response for Rectangular, L=' num2str(L)]);
grid on
% xlim([80 110])
% ylim([0.97 1.03])