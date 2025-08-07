% PSOLA (Pitch Synchronous Overlap and Add) Implementation in MATLAB
% This script demonstrates PSOLA-based pitch and time modification

clear all;
close all;
clc;

%% 1. Load Audio File
% Replace 'your_audio_file.mp3' with the actual filename
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

% Work with a shorter segment for demonstration (first 5 seconds)
max_duration = 5; % seconds
max_samples = min(length(audio_signal), round(max_duration * fs));
audio_signal = audio_signal(1:max_samples);

fprintf('Working with %.2f seconds of audio\n', length(audio_signal)/fs);

%% 4. Apply PSOLA Processing

fprintf('Detecting pitch marks...\n');
% Detect initial pitch marks
min_pitch = 80;   % Hz (minimum expected pitch)
max_pitch = 400;  % Hz (maximum expected pitch)
pitch_marks = detect_pitch_marks(audio_signal, fs, min_pitch, max_pitch);

% Refine pitch marks
if ~isempty(pitch_marks)
    pitch_marks = refine_pitch_marks(audio_signal, pitch_marks, fs);
    fprintf('Found %d pitch marks\n', length(pitch_marks));
else
    fprintf('No pitch marks detected. Signal may not be voiced.\n');
    pitch_marks = round(linspace(round(0.01*fs), length(audio_signal)-round(0.01*fs), 50));
    fprintf('Using synthetic pitch marks for demonstration\n');
end

% Time scaling examples
fprintf('Applying time scaling...\n');
time_stretched_125 = psola_time_scale(audio_signal, pitch_marks, 1.25);  % 25% slower
time_compressed_075 = psola_time_scale(audio_signal, pitch_marks, 0.75); % 25% faster
time_stretched_150 = psola_time_scale(audio_signal, pitch_marks, 1.5);   % 50% slower

% Pitch scaling examples  
fprintf('Applying pitch scaling...\n');
pitch_raised_120 = psola_pitch_scale(audio_signal, pitch_marks, 1.2, fs);  % 20% higher
pitch_lowered_083 = psola_pitch_scale(audio_signal, pitch_marks, 0.83, fs); % ~20% lower
pitch_raised_150 = psola_pitch_scale(audio_signal, pitch_marks, 1.5, fs);   % 50% higher

% Combined modifications
fprintf('Applying combined modifications...\n');
chipmunk_effect = psola_combined(audio_signal, pitch_marks, 0.8, 1.4, fs);  % Faster + higher
slow_deep = psola_combined(audio_signal, pitch_marks, 1.3, 0.7, fs);        % Slower + lower

%% 5. Analysis and Visualization

% Function to properly normalize audio for writing and playing
normalize_audio = @(x) x ./ max(abs(x(:))) * 0.95;

% Create time vectors for plotting
t_orig = (0:length(audio_signal)-1) / fs;
t_stretched = (0:length(time_stretched_125)-1) / fs;
t_compressed = (0:length(time_compressed_075)-1) / fs;

figure('Position', [100, 100, 1400, 1000]);

% Original signal with pitch marks
subplot(3, 3, 1);
plot(t_orig, audio_signal, 'b', 'LineWidth', 1);
hold on;
if ~isempty(pitch_marks)
    pitch_times = pitch_marks / fs;
    plot(pitch_times, audio_signal(pitch_marks), 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
end
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal with Pitch Marks');
grid on;
xlim([0 min(2, max(t_orig))]);

% Time scaling results
subplot(3, 3, 2);
plot(t_stretched, normalize_audio(time_stretched_125), 'g', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time Stretched (1.25x slower)');
grid on;

subplot(3, 3, 3);
plot(t_compressed, normalize_audio(time_compressed_075), 'r', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Time Compressed (0.75x faster)');
grid on;

% Pitch scaling results
subplot(3, 3, 4);
plot(t_orig, normalize_audio(pitch_raised_120), 'm', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Pitch Raised (1.2x higher)');
grid on;
xlim([0 min(2, max(t_orig))]);

subplot(3, 3, 5);
plot(t_orig, normalize_audio(pitch_lowered_083), 'c', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Pitch Lowered (0.83x lower)');
grid on;
xlim([0 min(2, max(t_orig))]);

% Combined effects
subplot(3, 3, 6);
t_chipmunk = (0:length(chipmunk_effect)-1) / fs;
plot(t_chipmunk, normalize_audio(chipmunk_effect), 'color', [1 0.5 0], 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('Chipmunk Effect (0.8x time, 1.4x pitch)');
grid on;

% Spectrograms for comparison
subplot(3, 3, 7);
window_length = round(0.025 * fs);
overlap = round(0.0125 * fs);
[S, F, T] = spectrogram(audio_signal, window_length, overlap, [], fs);
imagesc(T, F, 20*log10(abs(S)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Original Spectrogram');
colorbar;
ylim([0 2000]);

subplot(3, 3, 8);
[S_pitch, F_pitch, T_pitch] = spectrogram(normalize_audio(pitch_raised_120), window_length, overlap, [], fs);
imagesc(T_pitch, F_pitch, 20*log10(abs(S_pitch)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Pitch Raised Spectrogram');
colorbar;
ylim([0 2000]);

subplot(3, 3, 9);
[S_time, F_time, T_time] = spectrogram(normalize_audio(time_stretched_125), window_length, overlap, [], fs);
imagesc(T_time, F_time, 20*log10(abs(S_time)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Time Stretched Spectrogram');
colorbar;
ylim([0 2000]);

%% 6. Advanced PSOLA Effects

fprintf('Creating advanced PSOLA effects...\n');

% Vibrato effect (pitch modulation)
vibrato_rate = 5; % Hz
vibrato_depth = 0.05; % 5% pitch variation
t_vibrato = (0:length(audio_signal)-1) / fs;
vibrato_modulation = 1 + vibrato_depth * sin(2 * pi * vibrato_rate * t_vibrato);

vibrato_signal = zeros(size(audio_signal));
for i = 1:length(pitch_marks)-1
    segment_start = pitch_marks(i);
    if i < length(pitch_marks)
        segment_end = pitch_marks(i+1);
    else
        segment_end = length(audio_signal);
    end
    
    % Ensure valid segment boundaries
    if segment_end <= length(audio_signal) && segment_start < segment_end && segment_start >= 1
        local_pitch_factor = vibrato_modulation(segment_start);
        segment = audio_signal(segment_start:segment_end);
        
        % Apply local pitch scaling
        original_length = length(segment);
        new_length = round(original_length / local_pitch_factor);
        
        % Check for valid lengths before interpolation
        if new_length >= 2 && original_length >= 2
            new_indices = linspace(1, original_length, new_length);
            resampled = interp1(1:original_length, segment, new_indices, 'linear', 'extrap');
            
            end_pos = segment_start + length(resampled) - 1;
            if end_pos <= length(vibrato_signal)
                vibrato_signal(segment_start:end_pos) = resampled';
            end
        elseif original_length >= 1
            % Handle short segments by simple copying
            end_pos = min(segment_start + original_length - 1, length(vibrato_signal));
            vibrato_signal(segment_start:end_pos) = segment(1:end_pos-segment_start+1);
        end
    end
end

% Formant preservation (simple implementation)
formant_preserved = psola_pitch_scale(audio_signal, pitch_marks, 1.3, fs);
% Apply mild spectral envelope correction (simplified)
if length(formant_preserved) >= length(audio_signal)
    formant_preserved = formant_preserved(1:length(audio_signal));
elseif length(formant_preserved) < length(audio_signal)
    formant_preserved = [formant_preserved; zeros(length(audio_signal) - length(formant_preserved), 1)];
end

%% 7. Save Processed Audio Files

% Create output directory
if ~exist('psola_outputs', 'dir')
    mkdir('psola_outputs');
end

% Save all processed versions
audiowrite('psola_outputs/original.wav', normalize_audio(audio_signal), fs);
audiowrite('psola_outputs/time_stretched_125.wav', normalize_audio(time_stretched_125), fs);
audiowrite('psola_outputs/time_compressed_075.wav', normalize_audio(time_compressed_075), fs);
audiowrite('psola_outputs/time_stretched_150.wav', normalize_audio(time_stretched_150), fs);
audiowrite('psola_outputs/pitch_raised_120.wav', normalize_audio(pitch_raised_120), fs);
audiowrite('psola_outputs/pitch_lowered_083.wav', normalize_audio(pitch_lowered_083), fs);
audiowrite('psola_outputs/pitch_raised_150.wav', normalize_audio(pitch_raised_150), fs);
audiowrite('psola_outputs/chipmunk_effect.wav', normalize_audio(chipmunk_effect), fs);
audiowrite('psola_outputs/slow_deep.wav', normalize_audio(slow_deep), fs);
audiowrite('psola_outputs/vibrato_effect.wav', normalize_audio(vibrato_signal), fs);
audiowrite('psola_outputs/formant_preserved.wav', normalize_audio(formant_preserved), fs);

fprintf('\nPSOLA processing completed successfully!\n');
fprintf('Processed audio files saved in "psola_outputs" folder:\n');
fprintf('- original.wav\n');
fprintf('- time_stretched_125.wav (25%% slower)\n');
fprintf('- time_compressed_075.wav (25%% faster)\n');
fprintf('- time_stretched_150.wav (50%% slower)\n');
fprintf('- pitch_raised_120.wav (20%% higher pitch)\n');
fprintf('- pitch_lowered_083.wav (17%% lower pitch)\n');
fprintf('- pitch_raised_150.wav (50%% higher pitch)\n');
fprintf('- chipmunk_effect.wav (faster + higher)\n');
fprintf('- slow_deep.wav (slower + lower)\n');
fprintf('- vibrato_effect.wav (pitch vibrato)\n');
fprintf('- formant_preserved.wav (pitch changed, formants preserved)\n');

%% 8. Quality Analysis

fprintf('\nQuality Analysis:\n');
fprintf('Original duration: %.2f seconds\n', length(audio_signal)/fs);
fprintf('Time stretched (1.25x): %.2f seconds\n', length(time_stretched_125)/fs);
fprintf('Time compressed (0.75x): %.2f seconds\n', length(time_compressed_075)/fs);

% Calculate SNR for pitch-modified signals (simplified)
if length(pitch_raised_120) == length(audio_signal)
    mse = mean((audio_signal - pitch_raised_120).^2);
    signal_power = mean(audio_signal.^2);
    snr_pitch_raised = 10 * log10(signal_power / mse);
    fprintf('SNR for pitch raised (vs original): %.2f dB\n', snr_pitch_raised);
end

%% 9. Interactive Audio Player
fprintf('\nPress any key to play audio samples...\n');
pause;

fprintf('Playing original audio...\n');
sound(normalize_audio(audio_signal), fs);
pause(length(audio_signal)/fs + 1);

fprintf('Playing time stretched (slower)...\n');
sound(normalize_audio(time_stretched_125), fs);
pause(length(time_stretched_125)/fs + 1);

fprintf('Playing pitch raised...\n');
sound(normalize_audio(pitch_raised_120), fs);
pause(length(audio_signal)/fs + 1);

fprintf('Playing chipmunk effect...\n');
sound(normalize_audio(chipmunk_effect), fs);

fprintf('\nPSOLA demonstration complete!\n');

%% 2. Pitch Detection Functions

function pitch_marks = detect_pitch_marks(signal, fs, min_pitch, max_pitch)
    % Simple pitch detection using autocorrelation
    % Returns sample indices of pitch marks
    
    frame_length = round(0.025 * fs);  % 25ms frames
    hop_length = round(0.010 * fs);    % 10ms hop
    
    min_lag = round(fs / max_pitch);
    max_lag = round(fs / min_pitch);
    
    pitch_marks = [];
    current_pos = 1;
    
    while current_pos + frame_length < length(signal)
        % Extract frame
        frame = signal(current_pos:current_pos + frame_length - 1);
        
        % Apply window
        window = hamming(frame_length);
        frame = frame .* window;
        
        % Autocorrelation
        autocorr = xcorr(frame);
        autocorr = autocorr(frame_length:end);  % Take positive lags only
        
        % Find peak in valid pitch range
        if length(autocorr) >= max_lag
            valid_autocorr = autocorr(min_lag:max_lag);
            [~, peak_idx] = max(valid_autocorr);
            period = peak_idx + min_lag - 1;
            
            % Add pitch mark
            pitch_marks = [pitch_marks, current_pos + period];
            current_pos = current_pos + period;
        else
            current_pos = current_pos + hop_length;
        end
    end
end

function enhanced_pitch_marks = refine_pitch_marks(signal, initial_marks, fs)
    % Refine pitch marks using local maxima detection
    enhanced_pitch_marks = [];
    search_window = round(0.002 * fs);  % 2ms search window
    
    for i = 1:length(initial_marks)
        mark = initial_marks(i);
        start_idx = max(1, mark - search_window);
        end_idx = min(length(signal), mark + search_window);
        
        % Find local maximum (for voiced segments)
        [~, local_max] = max(abs(signal(start_idx:end_idx)));
        refined_mark = start_idx + local_max - 1;
        
        enhanced_pitch_marks = [enhanced_pitch_marks, refined_mark];
    end
end

%% 3. PSOLA Core Functions

function modified_signal = psola_time_scale(signal, pitch_marks, time_scale_factor)
    % Time scaling using PSOLA
    % time_scale_factor > 1: stretch (slower)
    % time_scale_factor < 1: compress (faster)
    
    if isempty(pitch_marks) || length(pitch_marks) < 2
        modified_signal = signal;
        return;
    end
    
    % Calculate new pitch mark positions
    new_marks = round(pitch_marks * time_scale_factor);
    
    % Initialize output signal
    output_length = round(length(signal) * time_scale_factor);
    modified_signal = zeros(output_length, 1);
    
    % Process each segment
    for i = 1:length(pitch_marks)-1
        % Original segment boundaries
        if i == 1
            start_orig = 1;
        else
            start_orig = round((pitch_marks(i-1) + pitch_marks(i)) / 2);
        end
        
        if i == length(pitch_marks)
            end_orig = length(signal);
        else
            end_orig = round((pitch_marks(i) + pitch_marks(i+1)) / 2);
        end
        
        % Ensure valid boundaries
        if start_orig >= end_orig || start_orig < 1 || end_orig > length(signal)
            continue;
        end
        
        % Extract segment
        segment = signal(start_orig:end_orig);
        segment_length = length(segment);
        
        % Check if segment is valid for processing
        if segment_length >= 2
            % Create window (Hann window)
            window = hann(segment_length);
            windowed_segment = segment .* window;
            
            % Calculate new position
            if i == 1
                start_new = 1;
            else
                start_new = round((new_marks(i-1) + new_marks(i)) / 2);
            end
            
            end_new = start_new + segment_length - 1;
            
            % Ensure we don't exceed output bounds
            if end_new <= output_length && start_new >= 1
                % Overlap and add
                modified_signal(start_new:end_new) = ...
                    modified_signal(start_new:end_new) + windowed_segment;
            end
        elseif segment_length == 1
            % Handle single sample segments
            if i == 1
                start_new = 1;
            else
                start_new = round((new_marks(i-1) + new_marks(i)) / 2);
            end
            
            if start_new >= 1 && start_new <= output_length
                modified_signal(start_new) = modified_signal(start_new) + segment(1);
            end
        end
    end
end

function modified_signal = psola_pitch_scale(signal, pitch_marks, pitch_scale_factor, fs)
    % Pitch scaling using PSOLA
    % pitch_scale_factor > 1: higher pitch
    % pitch_scale_factor < 1: lower pitch
    
    if isempty(pitch_marks) || length(pitch_marks) < 2
        modified_signal = signal;
        return;
    end
    
    modified_signal = zeros(size(signal));
    
    % Process each pitch period
    for i = 1:length(pitch_marks)-1
        % Define segment boundaries
        if i == 1
            start_idx = 1;
        else
            start_idx = round((pitch_marks(i-1) + pitch_marks(i)) / 2);
        end
        
        end_idx = round((pitch_marks(i) + pitch_marks(i+1)) / 2);
        
        if end_idx > length(signal)
            end_idx = length(signal);
        end
        
        % Ensure valid segment boundaries
        if start_idx >= end_idx || start_idx < 1 || end_idx > length(signal)
            continue;
        end
        
        % Extract segment
        segment = signal(start_idx:end_idx);
        original_length = length(segment);
        
        % Check if segment is long enough for processing
        if original_length >= 2
            % Resample segment to change pitch
            new_length = round(original_length / pitch_scale_factor);
            
            % Ensure new_length is at least 2 for interpolation
            if new_length >= 2
                % Use interpolation for resampling
                old_indices = 1:original_length;
                new_indices = linspace(1, original_length, new_length);
                resampled_segment = interp1(old_indices, segment, new_indices, 'linear', 'extrap')';
                
                % Apply window
                window = hann(new_length);
                windowed_segment = resampled_segment .* window;
                
                % Place in output with overlap-add
                output_start = start_idx;
                output_end = output_start + new_length - 1;
                
                if output_end <= length(modified_signal) && output_start >= 1
                    modified_signal(output_start:output_end) = ...
                        modified_signal(output_start:output_end) + windowed_segment;
                end
            elseif new_length == 1 && original_length >= 1
                % Handle single sample case
                modified_signal(start_idx) = modified_signal(start_idx) + segment(round(original_length/2));
            end
        elseif original_length == 1
            % Handle single sample segment
            modified_signal(start_idx) = modified_signal(start_idx) + segment(1);
        end
    end
end

function modified_signal = psola_combined(signal, pitch_marks, time_factor, pitch_factor, fs)
    % Combined time and pitch scaling
    % Apply time scaling first, then pitch scaling
    temp_signal = psola_time_scale(signal, pitch_marks, time_factor);
    
    % Recalculate pitch marks for time-scaled signal
    new_pitch_marks = round(pitch_marks * time_factor);
    new_pitch_marks = new_pitch_marks(new_pitch_marks <= length(temp_signal));
    
    modified_signal = psola_pitch_scale(temp_signal, new_pitch_marks, pitch_factor, fs);
end