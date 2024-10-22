% Analysis function analyzeSNAP, takes vector x and integer for sampling
% frequency Fs.
function [ampl,latency,amax,amin,tmin] = analyzeSNAP(x,Fs)
    [amax, tmax] = max(x);
    [amin, tmin] = min(x);
    ampl = amax - amin;
    latency = tmax - 1;
end