%% Frequency Modulation (FM) Synthesis: Basic Implementation
% Generate sound using frequency modulation - a nonlinear synthesis method
clear all; close all;

% Parameters
fs = 44100; % Sampling frequency (Hz)
duration = 3; % Duration of sound (seconds)
t = 0:1/fs:duration-1/fs; % Time vector

% FM synthesis parameters
carrier_freq = 440; % Carrier frequency in Hz (A4 note)
modulator_freq = 100; % Modulator frequency in Hz
modulation_index = 5; % Depth of frequency modulation (beta)
modulation_depth = modulation_index * modulator_freq; % Frequency deviation in Hz

% FM synthesis formula
modulator = sin(2 * pi * modulator_freq * t); % Modulator signal
carrier_phase = 2 * pi * carrier_freq * t + modulation_index * modulator; % Modulated phase
fm_signal = sin(carrier_phase); % FM synthesis result

% Normalize the signal
fm_signal = fm_signal / max(abs(fm_signal));

% Play sound
sound(fm_signal, fs);

% Visualization
figure;
subplot(3,1,1);
plot(t(1:1000), modulator(1:1000));
title('Modulator Signal');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(3,1,2);
plot(t(1:1000), fm_signal(1:1000));
title('FM Synthesis Waveform');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Frequency analysis
N = length(fm_signal);
f = (0:N-1)*(fs/N);
Y = fft(fm_signal);
Y = abs(Y(1:N/2));
f = f(1:N/2);

subplot(3,1,3);
plot(f, Y);
title('FM Synthesis: Frequency Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 2000]); grid on;

%% Advanced FM Synthesis with Multiple Operators
% Complex FM synthesis using multiple operators (like DX7 synthesizer)
clear variables; close all;

% Parameters
fs = 44100;
duration = 4;
t = 0:1/fs:duration-1/fs;

% Operator frequencies (frequency ratios are common in FM synthesis)
fundamental = 220; % Base frequency
op1_freq = fundamental * 1.0; % Operator 1 (1:1 ratio)
op2_freq = fundamental * 2.0; % Operator 2 (2:1 ratio)
op3_freq = fundamental * 3.5; % Operator 3 (3.5:1 ratio)
op4_freq = fundamental * 0.5; % Operator 4 (0.5:1 ratio - sub-harmonic)

% Modulation indices for each operator
mod_index_1 = 2.0;
mod_index_2 = 1.5;
mod_index_3 = 0.8;
mod_index_4 = 3.0;

% Generate operators
operator1 = sin(2 * pi * op1_freq * t);
operator2 = sin(2 * pi * op2_freq * t);
operator3 = sin(2 * pi * op3_freq * t);
operator4 = sin(2 * pi * op4_freq * t);

% Complex FM algorithm: Op4 -> Op3 -> Op2 -> Op1 (cascade configuration)
% Operator 4 modulates Operator 3
op3_modulated_phase = 2 * pi * op3_freq * t + mod_index_4 * operator4;
op3_modulated = sin(op3_modulated_phase);

% Operator 3 (modulated) modulates Operator 2
op2_modulated_phase = 2 * pi * op2_freq * t + mod_index_3 * op3_modulated;
op2_modulated = sin(op2_modulated_phase);

% Operator 2 (modulated) modulates Operator 1
op1_modulated_phase = 2 * pi * op1_freq * t + mod_index_2 * op2_modulated;
fm_complex = sin(op1_modulated_phase);

% Apply ADSR envelope
envelope = generateADSR(length(t), fs, 0.1, 0.3, 0.7, 1.5);
fm_complex = fm_complex .* envelope;

% Normalize and play
fm_complex = fm_complex / max(abs(fm_complex));
sound(fm_complex, fs);

% Visualization
figure;
subplot(2,2,1);
plot(t(1:2000), operator4(1:2000));
title('Operator 4 (Modulator)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(2,2,2);
plot(t(1:2000), op3_modulated(1:2000));
title('Operator 3 (Modulated by Op4)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(2,2,3);
plot(t(1:2000), op2_modulated(1:2000));
title('Operator 2 (Modulated by Op3)');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(2,2,4);
plot(t(1:2000), fm_complex(1:2000));
title('Final FM Output');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Frequency spectrum analysis
figure;
N = length(fm_complex);
f = (0:N-1)*(fs/N);
Y = fft(fm_complex);
Y = abs(Y(1:N/2));
f = f(1:N/2);

plot(f, Y);
title('Complex FM Synthesis: Frequency Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 3000]); grid on;

%% Time-Varying FM Synthesis
% FM synthesis with changing modulation index over time
clear all; close all;

% Parameters
fs = 44100;
duration = 5;
t = 0:1/fs:duration-1/fs;

% FM parameters
carrier_freq = 330; % Carrier frequency
modulator_freq = 220; % Modulator frequency

% Time-varying modulation index (creates evolving timbre)
mod_index_envelope = 0.5 + 4.5 * (1 - exp(-3*t)) .* exp(-0.5*t); % Attack and decay

% Generate time-varying FM signal
modulator_signal = sin(2 * pi * modulator_freq * t);
instantaneous_phase = 2 * pi * carrier_freq * t + mod_index_envelope .* modulator_signal;
fm_evolving = sin(instantaneous_phase);

% Apply amplitude envelope
amp_envelope = generateADSR(length(t), fs, 0.2, 0.8, 0.6, 2.0);
fm_evolving = fm_evolving .* amp_envelope;

% Normalize and play
fm_evolving = fm_evolving / max(abs(fm_evolving));
sound(fm_evolving, fs);

% Visualization
figure;
subplot(3,1,1);
plot(t(1:5000), mod_index_envelope(1:5000));
title('Modulation Index Envelope');
xlabel('Time (s)'); ylabel('Modulation Index'); grid on;

subplot(3,1,2);
plot(t(1:5000), fm_evolving(1:5000));
title('Time-Varying FM Synthesis');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Spectrogram to show frequency content evolution
subplot(3,1,3);
spectrogram(fm_evolving, hann(1024), 512, 1024, fs, 'yaxis');
title('Spectrogram: Frequency Evolution Over Time');

%% Parallel FM Synthesis (Multiple Carriers)
% FM synthesis with multiple parallel carrier-modulator pairs
clear variables; close all;

% Parameters
fs = 44100;
duration = 3;
t = 0:1/fs:duration-1/fs;

% Define multiple carrier-modulator pairs
num_pairs = 4;
carrier_freqs = [220, 330, 440, 550]; % Different carriers
modulator_freqs = [100, 150, 80, 200]; % Different modulators
mod_indices = [3, 2, 4, 1.5]; % Different modulation depths
amplitudes = [0.3, 0.25, 0.2, 0.25]; % Different mix levels

% Generate parallel FM signals
fm_parallel = zeros(size(t));

for i = 1:num_pairs
    % Generate modulator
    modulator = sin(2 * pi * modulator_freqs(i) * t);
    
    % Generate FM signal for this pair
    carrier_phase = 2 * pi * carrier_freqs(i) * t + mod_indices(i) * modulator;
    fm_component = amplitudes(i) * sin(carrier_phase);
    
    % Add to parallel sum
    fm_parallel = fm_parallel + fm_component;
end

% Apply global envelope
envelope = generateADSR(length(t), fs, 0.15, 0.4, 0.8, 1.0);
fm_parallel = fm_parallel .* envelope;

% Normalize and play
fm_parallel = fm_parallel / max(abs(fm_parallel));
sound(fm_parallel, fs);

% Visualization
figure;
subplot(2,1,1);
plot(t(1:2000), fm_parallel(1:2000));
title('Parallel FM Synthesis');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Show individual components
figure;
for i = 1:num_pairs
    subplot(2,2,i);
    modulator = sin(2 * pi * modulator_freqs(i) * t);
    carrier_phase = 2 * pi * carrier_freqs(i) * t + mod_indices(i) * modulator;
    fm_component = amplitudes(i) * sin(carrier_phase);
    
    plot(t(1:1000), fm_component(1:1000));
    title(sprintf('FM Pair %d: C=%.0fHz, M=%.0fHz', i, carrier_freqs(i), modulator_freqs(i)));
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
end

% Frequency analysis
N = length(fm_parallel);
f = (0:N-1)*(fs/N);
Y = fft(fm_parallel);
Y = abs(Y(1:N/2));
f = f(1:N/2);

figure;
plot(f, Y);
title('Parallel FM Synthesis: Frequency Spectrum');
xlabel('Frequency (Hz)'); ylabel('Magnitude');
xlim([0 2500]); grid on;

%% Feedback FM Synthesis
% FM synthesis where the output feeds back into itself
clear all; close all;

% Parameters
fs = 44100;
duration = 2;
t = 0:1/fs:duration-1/fs;

% Feedback FM parameters
carrier_freq = 440;
feedback_amount = 2.5; % Feedback modulation index

% Initialize signals
fm_feedback = zeros(size(t));
previous_sample = 0;

% Generate feedback FM signal (sample by sample)
for i = 1:length(t)
    % Current phase includes feedback from previous sample
    current_phase = 2 * pi * carrier_freq * t(i) + feedback_amount * previous_sample;
    
    % Generate current sample
    current_sample = sin(current_phase);
    fm_feedback(i) = current_sample;
    
    % Store for next iteration
    previous_sample = current_sample;
end

% Apply envelope
envelope = generateADSR(length(t), fs, 0.05, 0.2, 0.7, 0.8);
fm_feedback = fm_feedback .* envelope;

% Normalize and play
fm_feedback = fm_feedback / max(abs(fm_feedback));
sound(fm_feedback, fs);

% Visualization
figure;
subplot(2,1,1);
plot(t(1:2000), fm_feedback(1:2000));
title('Feedback FM Synthesis');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Frequency analysis
N = length(fm_feedback);
f = (0:N-1)*(fs/N);
Y = fft(fm_feedback);
Y = abs(Y(1:N/2));
f = f(1:N/2);

subplot(2,1,2);
semilogy(f, Y);
title('Feedback FM: Frequency Spectrum (Log Scale)');
xlabel('Frequency (Hz)'); ylabel('Magnitude (log)');
xlim([0 5000]); grid on;

%% FM Synthesis with Non-Sinusoidal Modulators
% Using different waveforms as modulators
clear variables; close all;

% Parameters
fs = 44100;
duration = 4;
t = 0:1/fs:duration-1/fs;

% Carrier parameters
carrier_freq = 300;
mod_freq = 50;
mod_index = 3;

% Generate different modulator waveforms
sine_mod = sin(2 * pi * mod_freq * t);
sawtooth_mod = sawtooth(2 * pi * mod_freq * t);
square_mod = square(2 * pi * mod_freq * t);
triangle_mod = sawtooth(2 * pi * mod_freq * t, 0.5);

% Generate FM signals with different modulators
fm_sine = sin(2 * pi * carrier_freq * t + mod_index * sine_mod);
fm_sawtooth = sin(2 * pi * carrier_freq * t + mod_index * sawtooth_mod);
fm_square = sin(2 * pi * carrier_freq * t + mod_index * square_mod);
fm_triangle = sin(2 * pi * carrier_freq * t + mod_index * triangle_mod);

% Create sequence of different modulator types
segment_length = length(t) / 4;
fm_sequence = [fm_sine(1:segment_length), ...
               fm_sawtooth(segment_length+1:2*segment_length), ...
               fm_square(2*segment_length+1:3*segment_length), ...
               fm_triangle(3*segment_length+1:end)];

% Apply envelope
envelope = generateADSR(length(fm_sequence), fs, 0.2, 0.5, 0.8, 1.5);
fm_sequence = fm_sequence .* envelope;

% Normalize and play
fm_sequence = fm_sequence / max(abs(fm_sequence));
sound(fm_sequence, fs);

% Visualization
figure;
subplot(2,2,1);
plot(t(1:1000), fm_sine(1:1000));
title('FM with Sine Modulator');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(2,2,2);
plot(t(1:1000), fm_sawtooth(1:1000));
title('FM with Sawtooth Modulator');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(2,2,3);
plot(t(1:1000), fm_square(1:1000));
title('FM with Square Modulator');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

subplot(2,2,4);
plot(t(1:1000), fm_triangle(1:1000));
title('FM with Triangle Modulator');
xlabel('Time (s)'); ylabel('Amplitude'); grid on;

% Frequency comparison
figure;
modulators = {sine_mod, sawtooth_mod, square_mod, triangle_mod};
fm_signals = {fm_sine, fm_sawtooth, fm_square, fm_triangle};
titles = {'Sine Mod', 'Sawtooth Mod', 'Square Mod', 'Triangle Mod'};

for i = 1:4
    subplot(2,2,i);
    N = length(fm_signals{i});
    f = (0:N-1)*(fs/N);
    Y = fft(fm_signals{i});
    Y = abs(Y(1:N/2));
    f = f(1:N/2);
    
    plot(f, Y);
    title(['FM Spectrum: ' titles{i}]);
    xlabel('Frequency (Hz)'); ylabel('Magnitude');
    xlim([0 2000]); grid on;
end

% ADSR envelope generator function
function env = generateADSR(num_samples, sample_rate, attack, decay, sustain, release)
    attack_samples = round(attack * sample_rate);
    decay_samples = round(decay * sample_rate);
    release_samples = round(release * sample_rate);
    sustain_samples = num_samples - attack_samples - decay_samples - release_samples;
    
    if sustain_samples < 0
        sustain_samples = round(0.1 * sample_rate);
        release_samples = num_samples - attack_samples - decay_samples - sustain_samples;
    end
    
    attack_phase = linspace(0, 1, attack_samples);
    decay_phase = linspace(1, sustain, decay_samples);
    sustain_phase = sustain * ones(1, sustain_samples);
    release_phase = linspace(sustain, 0, release_samples);
    
    env = [attack_phase, decay_phase, sustain_phase, release_phase];
    
    if length(env) > num_samples
        env = env(1:num_samples);
    elseif length(env) < num_samples
        env = [env, zeros(1, num_samples - length(env))];
    end
end