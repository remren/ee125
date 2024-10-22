n  = linspace(0, 50, 51);
x1 = (0.9 .^ n) .* cos(0.2 .* pi .* n);
x2 = cos(0.5*pi .* n);

L1 = 51;
L2 = 51;
% Length of the linear convolution
N = L1 + L2 - 1;

% Zero-pad the sequences to the same length N
X1 = fft(x1, N);
X2 = fft(x2, N);

% Multiply the DFTs
Y = X1 .* X2;

% Compute the inverse DFT to get the linear convolution
y_conv = ifft(Y);

% Plot the result
figure;
subplot(3,1,1);
stem(n, x1, '.');
title('Signal x_1[n]');
xlabel('n');
ylabel('x_1[n]');

subplot(3,1,2);
stem(n, x2, '.');
title('Signal x_2[n]');
xlabel('n');
ylabel('x_2[n]');

subplot(3,1,3);
n_conv = 0:N-1; % New n range for the convolution
stem(n_conv, real(y_conv), '.'); % Take only the real part of the convolution
title('Linear Convolution of x_1[n] and x_2[n]');
xlabel('n');
ylabel('y[n]');