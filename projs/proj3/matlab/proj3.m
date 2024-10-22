
% 1.) Plot the data
% All data is plotted at Fs = 10,000 Hz
% 12 ms data total, uV for data, time vector is in ms
t1 = 12; % in ms
Fs = 10000; % in Hz
t1_ls = linspace(0, t1, t1 * Fs);
t1_space = linspace(0, 12, 120);
plot(t1_space, cleanSnap_uV)
hold on
plot(t1_space, contamSnapBaseline_uV)
plot(t1_space, contamSnapPowerline_uV)
hold off
ylabel("Signal Output in uV")
xlabel("t, in ms")
title("SNAP Signal Comparison")
grid on
legend('cleanSnap', 'contamSnapBaseline', 'contamSnapPowerline')

% 2.) Create the analysis function - see analyzeSNAP.m
[amax, tmax] = max(cleanSnap_uV);

test = analyzeSNAP(cleanSnap_uV,Fs)
    
% 3.) Analyze reference SNAP
% 4.) Design a IIR highpass
% 5.) Design a FIR highpass
%       Discussion
%       Discussion
% 6.) Explore computational efficiency
%       Discussion
% 7.) Dilter the reference signal
% 7a.) Discussion
% 7b.) Discussion
% 8.) Filter the contaminated signals
% 8a.) 2x2 subplot
% 8b.) Discussion
% 9.) Investigate Baseline Removal technique
%       Discussion
% 10.) Remove phase distortion
%       Discussion
