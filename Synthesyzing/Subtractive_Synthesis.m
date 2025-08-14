%% Subtractive Synthesis: Basic Implementation
% Generate a sound by filtering a harmonically rich source
clear all; close all;

% Parameters
fs = 44100; % Sampling frequency (Hz)
duration = 3; % Duration of sound (seconds)
t = 0:1/fs:duration-1/fs; % Time vector

% Generate harmonically rich source (sawtooth wave)
fundamental = 220; % Base frequency (Hz)
source_signal = sawtooth(2 * pi * fundamental * t);

% Apply ADSR envelope
attack_time = 0.1;
decay_time = 0.3;
sustain_level = 0.6;
release_time = 0.5;

envelope = generateADSR(length(t), fs, attack_time, decay_time, sustain_level, release_time);
source_signal = source_signal .* envelope;

% Apply low-pass filter (subtractive synthesis)
cutoff_freq = 800; % Cutoff frequency (Hz)
filter_order = 4; % Filter order

% Design Butterworth low-pass filter
[b, a] = butter(filter_order, cutoff_freq/(fs/2), 'low');

% Apply filter to source signal
filtered_signal = filter(b, a, source_signal);

% Normalize signal to avoid clipping
filtered_signal = filtered_signal / max(abs(filtered_signal));

% Play sound
sound(filtered_signal, fs);

% Visualization
figure;
subplot(3,1,1);
plot(t(1:2000), source_signal(1:2000));
title('Original Sawtooth Wave (Harmonically Rich)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(3,1,2);
plot(t(1:2000), filtered_signal(1:2000));
title('Filtered Signal (Subtractive Synthesis)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(3,1,3);
plot(t(1:2000), envelope(1:2000));
title('ADSR Envelope');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Frequency domain analysis
figure;
% Original signal spectrum
N = length(source_signal);
f = (0:N-1)*(fs/N);
Y_original = fft(source_signal);
Y_original = abs(Y_original(1:N/2));
f = f(1:N/2);

% Filtered signal spectrum
Y_filtered = fft(filtered_signal);
Y_filtered = abs(Y_filtered(1:N/2));

subplot(2,1,1);
plot(f, Y_original);
title('Original Signal: Frequency Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 3000]); grid on;

subplot(2,1,2);
plot(f, Y_filtered);
title('Filtered Signal: Frequency Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 3000]); grid on;

% Function to generate ADSR envelope
function env = generateADSR(num_samples, sample_rate, attack, decay, sustain, release)
    % Calculate sample indices for each phase
    attack_samples = round(attack * sample_rate);
    decay_samples = round(decay * sample_rate);
    release_samples = round(release * sample_rate);
    sustain_samples = num_samples - attack_samples - decay_samples - release_samples;
    
    % Ensure positive sustain samples
    if sustain_samples < 0
        sustain_samples = round(0.1 * sample_rate); % Minimum sustain
        release_samples = num_samples - attack_samples - decay_samples - sustain_samples;
    end
    
    % Generate envelope phases
    attack_phase = linspace(0, 1, attack_samples);
    decay_phase = linspace(1, sustain, decay_samples);
    sustain_phase = sustain * ones(1, sustain_samples);
    release_phase = linspace(sustain, 0, release_samples);
    
    % Combine phases
    env = [attack_phase, decay_phase, sustain_phase, release_phase];
    
    % Ensure correct length
    if length(env) > num_samples
        env = env(1:num_samples);
    elseif length(env) < num_samples
        env = [env, zeros(1, num_samples - length(env))];
    end
end