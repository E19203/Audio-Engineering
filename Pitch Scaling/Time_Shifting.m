
filename = 'Music.mp3';

try
    [audio_signal, fs] = audioread(filename);
    fprintf('Successfully loaded: %s\n', filename);
    fprintf('Sample Rate: %d Hz\n', fs);
    fprintf('Duration: %.2f seconds\n', length(audio_signal)/fs);
    fprintf('Number of channels: %d\n', size(audio_signal, 2));
catch
    error('Could not load the audio file. Please check the filename and path.');
end

% Convert to mono if stereo
if size(audio_signal, 2) > 1
    audio_signal = mean(audio_signal, 2);
    fprintf('Converted stereo to mono\n');
end

% Normalize audio
audio_signal = audio_signal / max(abs(audio_signal));

%% 3. Apply Different Time Shifting Techniques

fprintf('Applying time shifting techniques...\n');

% Define delay times
delay_10ms = round(0.010 * fs);    % 10ms delay
delay_25ms = round(0.025 * fs);    % 25ms delay  
delay_50ms = round(0.050 * fs);    % 50ms delay
delay_100ms = round(0.100 * fs);   % 100ms delay
delay_fractional = 15.7;           % Fractional delay in samples

% 1. Fixed Delays
fprintf('Creating fixed delays...\n');
delayed_10ms = apply_fixed_delay(audio_signal, delay_10ms);
delayed_25ms = apply_fixed_delay(audio_signal, delay_25ms);
delayed_50ms = apply_fixed_delay(audio_signal, delay_50ms);
delayed_100ms = apply_fixed_delay(audio_signal, delay_100ms);

% 2. Fractional Delay
fprintf('Creating fractional delay...\n');
delayed_fractional = apply_fractional_delay(audio_signal, delay_fractional, fs);

% 3. Variable Delay (Chorus effect)
fprintf('Creating variable delay (chorus)...\n');
chorus_rate = 0.5;  % Hz
chorus_depth = 5;   % samples
chorus_delay_func = @(n) delay_25ms + chorus_depth * sin(2 * pi * chorus_rate * n / fs);
delayed_chorus = apply_variable_delay(audio_signal, chorus_delay_func, fs);

% 4. Flanging effect (shorter, faster modulation)
fprintf('Creating flanging effect...\n');
flanger_rate = 0.3;  % Hz
flanger_depth = 3;   % samples
flanger_base_delay = 2; % samples
flanger_delay_func = @(n) flanger_base_delay + flanger_depth * sin(2 * pi * flanger_rate * n / fs);
delayed_flanger = apply_variable_delay(audio_signal, flanger_delay_func, fs);

% 5. Multi-tap Delay (Echo effect)
fprintf('Creating multi-tap delay...\n');
tap_delays = [delay_50ms, delay_100ms, delay_100ms*2, delay_100ms*3];
tap_gains = [0.6, 0.4, 0.25, 0.15];
delayed_multitap = apply_multitap_delay(audio_signal, tap_delays, tap_gains);

% 6. Feedback Delay (Classic delay/echo)
fprintf('Creating feedback delay...\n');
delayed_feedback = apply_feedback_delay(audio_signal, delay_100ms, 0.4, 0.3);

% 7. Stereo Imaging Effects
fprintf('Creating stereo effects...\n');
% Haas effect (precedence effect) - slight delay between channels
haas_delay = round(0.002 * fs);  % 2ms delay
stereo_haas_left = audio_signal;
stereo_haas_right = apply_fixed_delay(audio_signal, haas_delay);
stereo_haas = [stereo_haas_left, stereo_haas_right(1:length(stereo_haas_left))];

% Ping-pong delay
pingpong_left = apply_multitap_delay(audio_signal, [delay_100ms, delay_100ms*3], [0.5, 0.3]);
pingpong_right = apply_multitap_delay(audio_signal, [delay_100ms*2, delay_100ms*4], [0.4, 0.2]);
% Make same length
min_length = min(length(pingpong_left), length(pingpong_right));
stereo_pingpong = [pingpong_left(1:min_length), pingpong_right(1:min_length)];

%% 4. Comb Filtering Effects

fprintf('Creating comb filtering effects...\n');

% Feedforward comb filter
comb_delay = round(0.005 * fs);  % 5ms delay
comb_gain = 0.7;
delayed_comb_signal = apply_fixed_delay(audio_signal, comb_delay);
% Match lengths for addition
min_len = min(length(audio_signal), length(delayed_comb_signal));
comb_feedforward = audio_signal(1:min_len) + comb_gain * delayed_comb_signal(1:min_len);

% Feedback comb filter (resonant effect)
comb_feedback = apply_feedback_delay(audio_signal, comb_delay, 0.6, 1.0);

%% 5. Analysis and Visualization

% Function to properly normalize audio
normalize_audio = @(x) x ./ max(abs(x(:))) * 0.95;

% Create time vectors
t_orig = (0:length(audio_signal)-1) / fs;
t_delayed_10ms = (0:length(delayed_10ms)-1) / fs;
t_delayed_100ms = (0:length(delayed_100ms)-1) / fs;

figure('Position', [100, 100, 1500, 1000]);

% Time domain plots
subplot(3, 4, 1);
plot(t_orig, audio_signal, 'b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal');
grid on;
xlim([0 min(3, max(t_orig))]);

subplot(3, 4, 2);
plot(t_delayed_10ms, normalize_audio(delayed_10ms), 'r', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('10ms Delay');
grid on;
xlim([0 min(3, max(t_delayed_10ms))]);

subplot(3, 4, 3);
plot(t_delayed_100ms, normalize_audio(delayed_100ms), 'g', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('100ms Delay');
grid on;
xlim([0 min(3, max(t_delayed_100ms))]);

subplot(3, 4, 4);
t_fractional = (0:length(delayed_fractional)-1) / fs;
plot(t_fractional, normalize_audio(delayed_fractional), 'm', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Fractional Delay (15.7 samples)');
grid on;
xlim([0 min(3, max(t_fractional))]);

% Variable delay effects
subplot(3, 4, 5);
t_chorus = (0:length(delayed_chorus)-1) / fs;
plot(t_chorus, normalize_audio(delayed_chorus), 'color', [1 0.5 0], 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Chorus Effect');
grid on;
xlim([0 min(3, max(t_chorus))]);

subplot(3, 4, 6);
t_flanger = (0:length(delayed_flanger)-1) / fs;
plot(t_flanger, normalize_audio(delayed_flanger), 'color', [0.5 0 1], 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Flanger Effect');
grid on;
xlim([0 min(3, max(t_flanger))]);

% Multi-tap and feedback delays
subplot(3, 4, 7);
t_multitap = (0:length(delayed_multitap)-1) / fs;
plot(t_multitap, normalize_audio(delayed_multitap), 'color', [0 0.7 0.7], 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Multi-tap Delay');
grid on;
xlim([0 min(5, max(t_multitap))]);

subplot(3, 4, 8);
t_feedback = (0:length(delayed_feedback)-1) / fs;
plot(t_feedback, normalize_audio(delayed_feedback), 'color', [0.7 0 0.7], 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Feedback Delay');
grid on;
xlim([0 min(5, max(t_feedback))]);

% Comb filtering effects
subplot(3, 4, 9);
t_comb_ff = (0:length(comb_feedforward)-1) / fs;
plot(t_comb_ff, normalize_audio(comb_feedforward), 'color', [0.8 0.4 0], 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Feedforward Comb Filter');
grid on;
xlim([0 min(3, max(t_comb_ff))]);

subplot(3, 4, 10);
t_comb_fb = (0:length(comb_feedback)-1) / fs;
plot(t_comb_fb, normalize_audio(comb_feedback), 'color', [0.4 0.8 0], 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Feedback Comb Filter');
grid on;
xlim([0 min(3, max(t_comb_fb))]);

% Frequency response analysis
subplot(3, 4, 11);
NFFT = 2^nextpow2(length(audio_signal));
f = fs/2 * linspace(0, 1, NFFT/2+1);

% Original spectrum
Y_orig = fft(audio_signal, NFFT);
magnitude_orig = abs(Y_orig(1:NFFT/2+1));

% Comb filter spectrum
Y_comb = fft(normalize_audio(comb_feedforward), NFFT);
magnitude_comb = abs(Y_comb(1:NFFT/2+1));

semilogx(f, 20*log10(magnitude_orig), 'b', 'LineWidth', 1.5);
hold on;
semilogx(f, 20*log10(magnitude_comb), 'r--', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Frequency Response: Original vs Comb Filter');
legend('Original', 'Comb Filtered', 'Location', 'best');
grid on;
xlim([20 fs/2]);

% Cross-correlation analysis (memory-efficient)
subplot(3, 4, 12);
% Use only a short segment for cross-correlation to save memory
segment_length = min(round(2 * fs), length(audio_signal));  % Max 2 seconds
audio_segment = audio_signal(1:segment_length);

% Get corresponding delayed segment
if length(delayed_50ms) >= segment_length
    delayed_segment = delayed_50ms(1:segment_length);
else
    delayed_segment = [delayed_50ms; zeros(segment_length - length(delayed_50ms), 1)];
end

% Limit cross-correlation lag range to save memory
max_lag = round(0.1 * fs);  % ±100ms maximum lag
[correlation, lags] = xcorr(audio_segment, delayed_segment, max_lag);
lag_time = lags / fs;

plot(lag_time, correlation, 'k', 'LineWidth', 1);
xlabel('Lag Time (s)');
ylabel('Cross-correlation');
title('Cross-correlation: Original vs Delayed (2s segment)');
grid on;
xlim([-0.1 0.1]);

%% 6. Advanced Time Shifting Effects

fprintf('Creating advanced effects...\n');

% Reverse delay (pre-echo effect)
reverse_delay = round(0.030 * fs);
if reverse_delay < length(audio_signal)
    reverse_delayed = [audio_signal(reverse_delay+1:end); zeros(reverse_delay, 1)];
else
    reverse_delayed = zeros(size(audio_signal));
end

%% 6. Advanced Time Shifting Effects

fprintf('Creating advanced effects...\n');

% Reverse delay (pre-echo effect)
reverse_delay = round(0.030 * fs);
if reverse_delay < length(audio_signal)
    reverse_delayed = [audio_signal(reverse_delay+1:end); zeros(reverse_delay, 1)];
else
    reverse_delayed = zeros(size(audio_signal));
end

% Granular delay (memory-optimized)
grain_size = round(0.050 * fs);  % 50ms grains
grain_overlap = round(0.025 * fs); % 25ms overlap
num_grains = floor((length(audio_signal) - grain_size) / grain_overlap) + 1;

% Limit number of grains to prevent memory issues
max_grains = min(num_grains, 15);
granular_delayed = zeros(length(audio_signal) + delay_100ms, 1);

for i = 1:max_grains
    start_idx = (i-1) * grain_overlap + 1;
    end_idx = start_idx + grain_size - 1;
    
    if end_idx <= length(audio_signal)
        grain = audio_signal(start_idx:end_idx);
        
        % Create window only if grain_size > 1
        if grain_size > 1
            window = hann(grain_size);
            windowed_grain = grain .* window;
        else
            windowed_grain = grain;
        end
        
        % Random delay for each grain
        grain_delay = round(rand() * delay_50ms);
        output_start = start_idx + grain_delay;
        output_end = output_start + length(windowed_grain) - 1;
        
        if output_end <= length(granular_delayed) && output_start >= 1
            granular_delayed(output_start:output_end) = ...
                granular_delayed(output_start:output_end) + 0.5 * windowed_grain;
        end
    end
end

%% 7. Save Processed Audio Files

% Create output directory
if ~exist('time_shifted_outputs', 'dir')
    mkdir('time_shifted_outputs');
end

% Save all processed versions
audiowrite('time_shifted_outputs/original.wav', normalize_audio(audio_signal), fs);
audiowrite('time_shifted_outputs/delayed_10ms.wav', normalize_audio(delayed_10ms), fs);
audiowrite('time_shifted_outputs/delayed_25ms.wav', normalize_audio(delayed_25ms), fs);
audiowrite('time_shifted_outputs/delayed_50ms.wav', normalize_audio(delayed_50ms), fs);
audiowrite('time_shifted_outputs/delayed_100ms.wav', normalize_audio(delayed_100ms), fs);
audiowrite('time_shifted_outputs/delayed_fractional.wav', normalize_audio(delayed_fractional), fs);
audiowrite('time_shifted_outputs/chorus_effect.wav', normalize_audio(delayed_chorus), fs);
audiowrite('time_shifted_outputs/flanger_effect.wav', normalize_audio(delayed_flanger), fs);
audiowrite('time_shifted_outputs/multitap_delay.wav', normalize_audio(delayed_multitap), fs);
audiowrite('time_shifted_outputs/feedback_delay.wav', normalize_audio(delayed_feedback), fs);
audiowrite('time_shifted_outputs/stereo_haas.wav', normalize_audio(stereo_haas), fs);
audiowrite('time_shifted_outputs/stereo_pingpong.wav', normalize_audio(stereo_pingpong), fs);
audiowrite('time_shifted_outputs/comb_feedforward.wav', normalize_audio(comb_feedforward), fs);
audiowrite('time_shifted_outputs/comb_feedback.wav', normalize_audio(comb_feedback), fs);
audiowrite('time_shifted_outputs/reverse_delayed.wav', normalize_audio(reverse_delayed), fs);
audiowrite('time_shifted_outputs/granular_delayed.wav', normalize_audio(granular_delayed), fs);

fprintf('\nTime shifting processing completed successfully!\n');
fprintf('Processed audio files saved in "time_shifted_outputs" folder:\n');
fprintf('- original.wav\n');
fprintf('- delayed_10ms.wav (10ms fixed delay)\n');
fprintf('- delayed_25ms.wav (25ms fixed delay)\n');
fprintf('- delayed_50ms.wav (50ms fixed delay)\n');
fprintf('- delayed_100ms.wav (100ms fixed delay)\n');
fprintf('- delayed_fractional.wav (fractional sample delay)\n');
fprintf('- chorus_effect.wav (modulated delay)\n');
fprintf('- flanger_effect.wav (short modulated delay)\n');
fprintf('- multitap_delay.wav (multiple echo taps)\n');
fprintf('- feedback_delay.wav (regenerative delay)\n');
fprintf('- stereo_haas.wav (stereo precedence effect)\n');
fprintf('- stereo_pingpong.wav (left-right alternating delay)\n');
fprintf('- comb_feedforward.wav (feedforward comb filtering)\n');
fprintf('- comb_feedback.wav (feedback comb filtering)\n');
fprintf('- reverse_delayed.wav (pre-echo effect)\n');
fprintf('- granular_delayed.wav (granular processing)\n');

%% 8. Performance Analysis

fprintf('\nPerformance Analysis:\n');
fprintf('Original signal length: %d samples (%.2f seconds)\n', ...
    length(audio_signal), length(audio_signal)/fs);

delays_analysis = [delay_10ms, delay_25ms, delay_50ms, delay_100ms];
delay_times = [10, 25, 50, 100];

for i = 1:length(delays_analysis)
    fprintf('%.0fms delay adds %d samples (%.3f seconds)\n', ...
        delay_times(i), delays_analysis(i), delays_analysis(i)/fs);
end

% Memory usage estimation
original_size = length(audio_signal) * 8;  % bytes (double precision)
delayed_size = length(delayed_multitap) * 8;
fprintf('Memory usage: Original %.1f KB, Multi-tap delay %.1f KB\n', ...
    original_size/1024, delayed_size/1024);

%% 9. Interactive Audio Player
fprintf('\nPress any key to play audio samples...\n');
pause;

fprintf('Playing original audio...\n');
sound(normalize_audio(audio_signal), fs);
pause(length(audio_signal)/fs + 1);

fprintf('Playing 50ms delayed audio...\n');
sound(normalize_audio(delayed_50ms), fs);
pause(3);

fprintf('Playing chorus effect...\n');
sound(normalize_audio(delayed_chorus), fs);
pause(3);

fprintf('Playing feedback delay...\n');
sound(normalize_audio(delayed_feedback), fs);

fprintf('\nTime shifting demonstration complete!\n');

%% 2. Time Shifting Functions

function delayed_signal = apply_fixed_delay(signal, delay_samples)
    % Apply fixed delay by padding with zeros
    if delay_samples > 0
        delayed_signal = [zeros(delay_samples, 1); signal];
    elseif delay_samples < 0
        % Advance (remove samples from beginning)
        abs_delay = abs(delay_samples);
        if abs_delay < length(signal)
            delayed_signal = signal(abs_delay + 1:end);
        else
            delayed_signal = zeros(size(signal));
        end
    else
        delayed_signal = signal;
    end
end

function delayed_signal = apply_fractional_delay(signal, delay_samples, fs)
    % Apply fractional delay using interpolation
    if abs(delay_samples) < 1e-6
        delayed_signal = signal;
        return;
    end
    
    % Separate integer and fractional parts
    integer_delay = floor(delay_samples);
    fractional_delay = delay_samples - integer_delay;
    
    if fractional_delay == 0
        % Pure integer delay
        delayed_signal = apply_fixed_delay(signal, integer_delay);
    else
        % Apply fractional delay using interpolation
        % Create time vectors
        original_time = (0:length(signal)-1)';
        shifted_time = original_time - delay_samples;
        
        % Interpolate
        valid_indices = (shifted_time >= 0) & (shifted_time <= length(signal)-1);
        delayed_signal = zeros(size(signal));
        
        if any(valid_indices)
            delayed_signal(valid_indices) = interp1(original_time, signal, ...
                shifted_time(valid_indices), 'linear', 0);
        end
        
        % Add integer delay if needed
        if integer_delay > 0
            delayed_signal = [zeros(integer_delay, 1); delayed_signal];
        elseif integer_delay < 0
            abs_delay = abs(integer_delay);
            if abs_delay < length(delayed_signal)
                delayed_signal = delayed_signal(abs_delay + 1:end);
            end
        end
    end
end

function output_signal = apply_variable_delay(signal, delay_function, fs)
    % Apply time-varying delay
    % delay_function: function handle that returns delay in samples for each sample
    
    output_signal = zeros(size(signal));
    
    for n = 1:length(signal)
        delay_samples = delay_function(n);
        source_index = n - delay_samples;
        
        % Interpolate if source index is valid and not integer
        if source_index >= 1 && source_index <= length(signal)
            if source_index == round(source_index)
                % Integer index
                output_signal(n) = signal(round(source_index));
            else
                % Fractional index - interpolate
                idx_floor = floor(source_index);
                idx_ceil = ceil(source_index);
                frac = source_index - idx_floor;
                
                if idx_floor >= 1 && idx_ceil <= length(signal)
                    output_signal(n) = (1 - frac) * signal(idx_floor) + ...
                                      frac * signal(idx_ceil);
                end
            end
        end
    end
end

function multitap_output = apply_multitap_delay(signal, delays, gains)
    % Apply multiple delays with different gains
    % delays: array of delay times in samples
    % gains: array of gain values for each delay
    
    if length(delays) ~= length(gains)
        error('Number of delays must match number of gains');
    end
    
    % Find maximum delay to determine output length
    max_delay = max(delays);
    output_length = length(signal) + max_delay;
    multitap_output = zeros(output_length, 1);
    
    % Add original signal
    multitap_output(1:length(signal)) = signal;
    
    % Add each delayed version
    for i = 1:length(delays)
        if delays(i) > 0 && gains(i) ~= 0
            delayed_version = apply_fixed_delay(signal, delays(i));
            % Ensure same length
            if length(delayed_version) > output_length
                delayed_version = delayed_version(1:output_length);
            elseif length(delayed_version) < output_length
                delayed_version = [delayed_version; zeros(output_length - length(delayed_version), 1)];
            end
            
            multitap_output = multitap_output + gains(i) * delayed_version;
        end
    end
end

function feedback_output = apply_feedback_delay(signal, delay_samples, feedback_gain, mix_level)
    % Apply delay with feedback
    % delay_samples: delay time in samples
    % feedback_gain: amount of delayed signal fed back (0-1)
    % mix_level: dry/wet mix (0=dry, 1=wet)
    
    if delay_samples <= 0
        feedback_output = signal;
        return;
    end
    
    % Initialize output and delay buffer
    output_length = length(signal) + delay_samples;
    feedback_output = zeros(output_length, 1);
    delay_buffer = zeros(delay_samples, 1);
    
    for n = 1:length(signal)
        % Get delayed sample from buffer
        delayed_sample = delay_buffer(1);
        
        % Input + feedback
        input_sample = signal(n) + feedback_gain * delayed_sample;
        
        % Shift delay buffer and add new sample
        delay_buffer = [delay_buffer(2:end); input_sample];
        
        % Output mix
        feedback_output(n) = (1 - mix_level) * signal(n) + mix_level * delayed_sample;
    end
    
    % Add remaining delay buffer to output
    for n = length(signal)+1:output_length
        delayed_sample = delay_buffer(1);
        delay_buffer = [delay_buffer(2:end); 0];
        feedback_output(n) = mix_level * delayed_sample;
    end
end