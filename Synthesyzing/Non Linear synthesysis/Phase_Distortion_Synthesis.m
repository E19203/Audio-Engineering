
% Parameters
fs = 44100; % Sampling frequency
duration = 3; % Duration in seconds
t = 0:1/fs:duration-1/fs; % Time vector

% Phase distortion synthesis parameters
carrier_freq = 440; % Carrier frequency in Hz
modulation_depth = 1.0; % Depth of phase distortion

% Phase distortion formula
phase = linspace(0, 2 * pi * carrier_freq * duration, length(t)); % Base phase
distorted_phase = sin(phase .* (1 + modulation_depth * sin(2 * pi * carrier_freq * t))); % Distorted phase

% Generate the signal using the distorted phase
pd_signal = sin(distorted_phase);

% Normalize the signal
pd_signal = pd_signal / max(abs(pd_signal));

% Play sound
sound(pd_signal, fs);

% Plot the result
figure;
subplot(2,1,1);
plot_samples = round(0.025 * fs);
plot(t(1:plot_samples), pd_signal(1:plot_samples)); % Plot the first samples
title('Phase Distortion Synthesis Waveform');
xlabel('Time [s]');
ylabel('Amplitude');
grid on;
xlim([0 0.025]);

% Plot frequency spectrum
N = length(pd_signal);
f = (0:N-1)*(fs/N);
Y = fft(pd_signal);
Y = abs(Y(1:N/2));
f = f(1:N/2);

subplot(2,1,2);
plot(f, Y);
title('Phase Distortion Synthesis: Frequency Spectrum');
xlabel('Frequency [Hz]');
ylabel('Magnitude');
xlim([0 2000]);
grid on;
