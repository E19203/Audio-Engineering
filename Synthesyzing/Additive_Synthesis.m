%% Alternative Implementation 1: Loop-based Approach
% Linear Synthesis: Additive Synthesis - Iterative summation
clc; clear; close all;

% System configuration
sample_rate = 44100;
sound_length = 2;
time_vector = 0:1/sample_rate:sound_length-1/sample_rate;

% Harmonic series definition
frequencies = [440, 880, 1320]; % Hz
amplitudes = [0.5, 0.3, 0.2];

% Initialize output signal
composite_signal = zeros(size(time_vector));

% Additive synthesis using loop
for harmonic = 1:length(frequencies)
    harmonic_component = amplitudes(harmonic) * sin(2 * pi * frequencies(harmonic) * time_vector);
    composite_signal = composite_signal + harmonic_component;
end

% Signal conditioning
composite_signal = composite_signal / max(abs(composite_signal));

% Audio output
sound(composite_signal, sample_rate);

% Visualization
figure;
subplot(2,1,1);
plot(time_vector(1:1000), composite_signal(1:1000));
title('Additive Synthesis: Time-Domain Waveform');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Spectral analysis
N = length(composite_signal);
f = (0:N-1)*(sample_rate/N);
Y = fft(composite_signal);
Y = abs(Y(1:N/2));
f = f(1:N/2);

subplot(2,1,2);
plot(f, Y);
title('Additive Synthesis: Frequency Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 2000]); grid on;
