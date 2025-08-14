%% FM Synthesis: Simple Implementation
clear all; close all;

% Parameters
fs = 44100; % Sampling frequency
duration = 3; % Duration in seconds
t = 0:1/fs:duration-1/fs; % Time vector

% FM synthesis parameters
carrier_freq = 440; % Carrier frequency in Hz
modulator_freq = 100; % Modulator frequency in Hz
modulation_index = 5; % Depth of frequency modulation (beta)

% FM synthesis formula
modulator = sin(2 * pi * modulator_freq * t); % Modulator signal
carrier_phase = 2 * pi * carrier_freq * t + modulation_index * modulator; % Modulated phase
fm_signal = sin(carrier_phase); % FM synthesis result

% Normalize the signal
fm_signal = fm_signal / max(abs(fm_signal));

% Play sound
sound(fm_signal, fs);

% Plot the result
figure;
subplot(2,1,1);
% Plot first 0.025 seconds (about 1100 samples) to match your image
plot_samples = round(0.025 * fs);
plot(t(1:plot_samples), fm_signal(1:plot_samples));
title('FM Synthesis Waveform');
xlabel('Time [s]');
ylabel('Amplitude');
grid on;
xlim([0 0.025]);

% Plot frequency spectrum
N = length(fm_signal);
f = (0:N-1)*(fs/N);
Y = fft(fm_signal);
Y = abs(Y(1:N/2));
f = f(1:N/2);

subplot(2,1,2);
plot(f, Y);
title('FM Synthesis: Frequency Spectrum');
xlabel('Frequency [Hz]');
ylabel('Magnitude');
xlim([0 2000]);
grid on;