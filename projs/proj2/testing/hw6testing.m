x = ones(1, 128);
N = 1024;

X = fft(x, N);
stem(0:N-1, X, '.');
xlim tight;
xlabel('DFT Sample k');
ylabel('|X[k]|');
title("N = " + N);