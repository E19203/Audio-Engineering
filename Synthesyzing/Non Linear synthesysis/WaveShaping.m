%% Wave Shaping Distortion Synthesis: Simple Implementation
clear all; close all;

% Parameters
fs = 44100; % Sampling frequency
duration = 3; % Duration in seconds
t = 0:1/fs:duration-1/fs; % Time vector
carrier_freq = 440; % Carrier frequency in Hz

% Parameters for wave shaping
input_wave = sin(2 * pi * carrier_freq * t); % Simple sine wave

% Apply wave shaping
drive = 5; % Drive parameter
shaped_signal = waveshaper(input_wave, drive);

% Normalize the signal
shaped_signal = shaped_signal / max(abs(shaped_signal));

% Play sound
sound(shaped_signal, fs);

% Plot the result
figure;
subplot(2,1,1);
plot_samples = round(0.025 * fs);
plot(t(1:plot_samples), shaped_signal(1:plot_samples)); % Plot the first samples
title('Wave Shaping Synthesis Waveform');
xlabel('Time [s]');
ylabel('Amplitude');
grid on;
xlim([0 0.025]);

% Plot frequency spectrum
N = length(shaped_signal);
f = (0:N-1)*(fs/N);
Y = fft(shaped_signal);
Y = abs(Y(1:N/2));
f = f(1:N/2);

subplot(2,1,2);
plot(f, Y);
title('Wave Shaping Synthesis: Frequency Spectrum');
xlabel('Frequency [Hz]');
ylabel('Magnitude');
xlim([0 2000]);
grid on;

% Wave shaping function definition (must be at end)
function output = waveshaper(input_wave, drive)
    output = sign(input_wave) .* (1 - exp(-drive * abs(input_wave)));
end