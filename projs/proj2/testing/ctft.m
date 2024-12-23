function [X,f] = ctft(x,fs)
%CTFT calculates the continuous-time Fourier transform (CTFT) 
% of a periodic signal x(t) which is reconstructed from the samples in
% the vector x using ideal bandlimited interpolation. The vector x
% contains samples of x(t) over an integer number of periods, and fs
% is the sampling frequency fs
%
% Usage: [X,f] = ctft(x,fs)
% 
% The vector X contains the (complex) samples of the continuous-time
% Fourier transform evaluated at the frequencies in the vector f.
% 
% This function makes make use of the relationship between the CTFT
% of x(t) and the DTFT of its samples x[n], as well as the
% relationship between the DTFT and the DFT of x[n].

% This function is a simple adaptation of:
%---------------------------------------------------------------
% copyright 1996, by John Buck, Michael Daniel, and Andrew Singer.
% For use with the textbook "Computer Explorations in Signals and
% Systems using MATLAB", Prentice Hall, 1997.
%---------------------------------------------------------------

N = numel(x);
X = fftshift(fft(x,N))*(2*pi/N);  % do fft; scale; shift so DC is in the middle
f = fs/N * (-floor(N/2) : floor((N-1)/2));
