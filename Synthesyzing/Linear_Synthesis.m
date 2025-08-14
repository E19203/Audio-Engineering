%% Alternative Implementation 1: Using Anonymous Functions
% Linear Synthesis: Amplitude Modulation (AM) - Function-based approach
clc; clear; close all;

% System parameters
sampling_rate = 44100;
sound_duration = 2;
time_axis = linspace(0, sound_duration, sampling_rate * sound_duration);

% Signal parameters
carrier_freq = 440;
modulator_freq = 5;
carrier_amp = 0.5;
modulation_depth = 0.8;

% Define signal components using anonymous functions
carrier_wave = @(t) cos(2 * pi * carrier_freq * t);
modulator_wave = @(t) cos(2 * pi * modulator_freq * t);
envelope = @(t) carrier_amp * (1 + modulation_depth * modulator_wave(t));

% Generate modulated signal
am_signal = envelope(time_axis) .* carrier_wave(time_axis);

% Normalize to prevent clipping
am_signal = am_signal ./ max(abs(am_signal));

% Audio playback
sound(am_signal, sampling_rate);

% Visualization
figure;
subplot(2,1,1);
plot(time_axis(1:1000), am_signal(1:1000));
title('Amplitude Modulation: Time-Domain Waveform');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Frequency analysis
signal_length = length(am_signal);
freq_vector = (0:signal_length-1) * (sampling_rate/signal_length);
spectrum = fft(am_signal);
magnitude_spectrum = abs(spectrum(1:signal_length/2));
freq_vector = freq_vector(1:signal_length/2);

subplot(2,1,2);
plot(freq_vector, magnitude_spectrum);
title('Amplitude Modulation: Frequency Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 1000]); grid on;