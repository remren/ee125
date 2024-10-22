% T  = 2;
Fs = 8192;

omega = 2 * pi * 1500;
B     = 2 * pi * 3000;

% t = 0:(1/Fs):T;
% 
% x = sin(omega * t + 0.5 * B * (t .^ 2));

t1 = 0:(1/Fs):0.5;
t2 = 0:(1/Fs):0.8;
t3 = 0:(1/Fs):1.2;
t4 = 0:(1/Fs):2;

x1 = sin(omega * t1 + 0.5 * B * (t1 .^ 2));
x2 = sin(omega * t2 + 0.5 * B * (t2 .^ 2));
x3 = sin(omega * t3 + 0.5 * B * (t3 .^ 2));
x4 = sin(omega * t4 + 0.5 * B * (t4 .^ 2));

[x1ctft,x1f] = ctft(x1,Fs);
[x2ctft,x2f] = ctft(x2,Fs);
[x3ctft,x3f] = ctft(x3,Fs);
[x4ctft,x4f] = ctft(x4,Fs);

% Plotting
figure

subplot(2, 2, 1) % 2 rows, 2 col, plot 1
plot(x1f, abs(x1ctft))
xlabel('Frequency, f in Hz')
ylabel('|Amplitude|')
title('Chirp Frequency Domain for 0.5s')

subplot(2, 2, 2) % 2 rows, 2 col, plot 2
plot(x2f, abs(x2ctft))
xlabel('Frequency, f in Hz')
ylabel('|Amplitude|')
title('Chirp Frequency Domain for 0.8s')

subplot(2, 2, 3) % 2 rows, 2 col, plot 3
plot(x3f, abs(x3ctft))
xlabel('Frequency, f in Hz')
ylabel('|Amplitude|')
title('Chirp Frequency Domain for 1.2s')

subplot(2, 2, 4) % 2 rows, 2 col, plot 4
plot(x4f, abs(x4ctft))
xlabel('Frequency, f in Hz')
ylabel('|Amplitude|')
title('Chirp Frequency Domain for 2s')
