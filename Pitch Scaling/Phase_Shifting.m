
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

%% 2. Design Phase Shift Filters

% Method 1: All-Pass Filter for constant phase shift
% Create all-pass filter with 90-degree phase shift
phase_shift_deg = 90;  % Phase shift in degrees
phase_shift_rad = phase_shift_deg * pi / 180;  % Convert to radians

% Design all-pass filter coefficients for 90-degree phase shift
% Using a simple all-pass filter: H(z) = (z^-1 - a) / (1 - a*z^-1)
a = 0.9;  % All-pass filter coefficient (affects phase response)
b = [a, -1];  % Numerator coefficients
a_coeff = [1, -a];  % Denominator coefficients

%% 3. Apply Phase Shifting Techniques

% Method 1: All-Pass Filter Phase Shift
phase_shifted_allpass = filter(b, a_coeff, audio_signal);

% Method 2: Hilbert Transform (90-degree phase shift)
phase_shifted_hilbert = imag(hilbert(audio_signal));

% Method 3: FFT-based Phase Shift
N = length(audio_signal);
X = fft(audio_signal);
frequencies = (0:N-1) * fs / N;

% Apply linear phase shift in frequency domain
phase_shift_samples = round(0.001 * fs);  % 1ms delay
phase_shift_vector = exp(-1j * 2 * pi * frequencies * phase_shift_samples / fs);
X_phase_shifted = X .* phase_shift_vector';
phase_shifted_fft = real(ifft(X_phase_shifted));

% Method 4: Variable Phase Shift (frequency-dependent)
% Create different phase shifts for different frequency bands
freq_bands = [0, 500, 1000, 2000, 4000, fs/2];  % Frequency band edges
phase_shifts = [0, 30, 60, 90, 45, 0];  % Phase shifts for each band (degrees)

X_variable = X;
for i = 1:length(freq_bands)-1
    % Find frequency indices for current band
    freq_start = freq_bands(i);
    freq_end = freq_bands(i+1);
    idx_start = round(freq_start * N / fs) + 1;
    idx_end = round(freq_end * N / fs) + 1;
    
    % Apply phase shift to this frequency band
    phase_shift_band = phase_shifts(i) * pi / 180;  % Convert to radians
    if idx_end <= length(frequencies)
        band_freqs = frequencies(idx_start:idx_end);
        phase_vector = exp(1j * phase_shift_band * ones(size(band_freqs)));
        X_variable(idx_start:idx_end) = X_variable(idx_start:idx_end) .* phase_vector';
        
        % Mirror for negative frequencies (for real signal)
        if idx_start > 1
            neg_idx_start = N - idx_end + 2;
            neg_idx_end = N - idx_start + 2;
            if neg_idx_end <= N
                X_variable(neg_idx_start:neg_idx_end) = conj(X_variable(idx_end:-1:idx_start));
            end
        end
    end
end
phase_shifted_variable = real(ifft(X_variable));

%% 4. Create Composite Effects

% Stereo Phase Effect (creates wide stereo image)
stereo_left = audio_signal;
stereo_right = phase_shifted_hilbert;
stereo_effect = [stereo_left, stereo_right];

% Chorus Effect using multiple phase shifts
chorus_delays = [0.002, 0.005, 0.008];  % Different delays in seconds
chorus_mix = audio_signal;
for delay_time = chorus_delays
    delay_samples = round(delay_time * fs);
    delayed_signal = [zeros(delay_samples, 1); audio_signal(1:end-delay_samples)];
    % Add slight phase shift to each delay
    delayed_phase_shifted = imag(hilbert(delayed_signal));
    chorus_mix = chorus_mix + 0.3 * delayed_phase_shifted;
end
chorus_mix = chorus_mix / max(abs(chorus_mix));  % Normalize

%% 5. Analysis and Visualization

% Time domain comparison
t = (0:length(audio_signal)-1) / fs;
sample_range = 1:min(4000, length(audio_signal));  % Show first 4000 samples

figure('Position', [100, 100, 1200, 800]);

% Time domain plots
subplot(3, 2, 1);
plot(t(sample_range), audio_signal(sample_range), 'b', 'LineWidth', 1.5);
hold on;
plot(t(sample_range), phase_shifted_hilbert(sample_range), 'r', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original vs Hilbert Phase Shift');
legend('Original', 'Phase Shifted', 'Location', 'best');
grid on;

subplot(3, 2, 2);
plot(t(sample_range), audio_signal(sample_range), 'b', 'LineWidth', 1.5);
hold on;
plot(t(sample_range), phase_shifted_allpass(sample_range), 'g', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original vs All-Pass Phase Shift');
legend('Original', 'All-Pass Shifted', 'Location', 'best');
grid on;

% Frequency domain analysis
NFFT = 2^nextpow2(length(audio_signal));
f = fs/2 * linspace(0, 1, NFFT/2+1);

% Original spectrum
Y_orig = fft(audio_signal, NFFT);
magnitude_orig = abs(Y_orig(1:NFFT/2+1));
phase_orig = angle(Y_orig(1:NFFT/2+1));

% Phase shifted spectrum
Y_shifted = fft(phase_shifted_hilbert, NFFT);
magnitude_shifted = abs(Y_shifted(1:NFFT/2+1));
phase_shifted_spectrum = angle(Y_shifted(1:NFFT/2+1));

subplot(3, 2, 3);
semilogx(f, 20*log10(magnitude_orig), 'b', 'LineWidth', 1.5);
hold on;
semilogx(f, 20*log10(magnitude_shifted), 'r--', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Magnitude Spectrum Comparison');
legend('Original', 'Phase Shifted', 'Location', 'best');
grid on;
xlim([20 fs/2]);

subplot(3, 2, 4);
semilogx(f, phase_orig, 'b', 'LineWidth', 1.5);
hold on;
semilogx(f, phase_shifted_spectrum, 'r--', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
title('Phase Spectrum Comparison');
legend('Original', 'Phase Shifted', 'Location', 'best');
grid on;
xlim([20 fs/2]);

% Phase difference plot
subplot(3, 2, 5);
phase_diff = phase_shifted_spectrum - phase_orig;
% Unwrap phase difference
phase_diff = unwrap(phase_diff);
semilogx(f, phase_diff, 'k', 'LineWidth', 2);
xlabel('Frequency (Hz)');
ylabel('Phase Difference (radians)');
title('Phase Difference between Original and Processed');
grid on;
xlim([20 fs/2]);

% Spectrogram comparison
subplot(3, 2, 6);
window_length = round(0.025 * fs);  % 25ms window
overlap = round(0.0125 * fs);       % 50% overlap
[S, F, T] = spectrogram(audio_signal, window_length, overlap, [], fs);
imagesc(T, F, 20*log10(abs(S)));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Original Signal Spectrogram');
colorbar;
colormap('jet');

%% 6. Save Processed Audio Files

% Create output directory if it doesn't exist
if ~exist('phase_shifted_outputs', 'dir')
    mkdir('phase_shifted_outputs');
end

% Save different processed versions
audiowrite('phase_shifted_outputs/original.wav', audio_signal, fs);
audiowrite('phase_shifted_outputs/hilbert_phase_shift.wav', ...
           phase_shifted_hilbert/max(abs(phase_shifted_hilbert)), fs);
audiowrite('phase_shifted_outputs/allpass_phase_shift.wav', ...
           phase_shifted_allpass/max(abs(phase_shifted_allpass)), fs);
audiowrite('phase_shifted_outputs/fft_phase_shift.wav', ...
           phase_shifted_fft/max(abs(phase_shifted_fft)), fs);
audiowrite('phase_shifted_outputs/variable_phase_shift.wav', ...
           phase_shifted_variable/max(abs(phase_shifted_variable)), fs);
audiowrite('phase_shifted_outputs/stereo_effect.wav', stereo_effect, fs);
audiowrite('phase_shifted_outputs/chorus_effect.wav', chorus_mix, fs);

fprintf('\nProcessing completed successfully!\n');
fprintf('Processed audio files saved in "phase_shifted_outputs" folder:\n');
fprintf('- original.wav\n');
fprintf('- hilbert_phase_shift.wav (90° phase shift)\n');
fprintf('- allpass_phase_shift.wav (all-pass filter)\n');
fprintf('- fft_phase_shift.wav (FFT-based delay)\n');
fprintf('- variable_phase_shift.wav (frequency-dependent)\n');
fprintf('- stereo_effect.wav (stereo phase effect)\n');
fprintf('- chorus_effect.wav (chorus with phase shifts)\n');

%% 7. Interactive Audio Player (Optional)
fprintf('\nPress any key to play audio samples...\n');
pause;

fprintf('Playing original audio...\n');
sound(audio_signal, fs);
pause(3);

fprintf('Playing Hilbert phase-shifted audio...\n');
sound(phase_shifted_hilbert/max(abs(phase_shifted_hilbert)), fs);
pause(3);

fprintf('Playing all-pass phase-shifted audio...\n');
sound(phase_shifted_allpass/max(abs(phase_shifted_allpass)), fs);

fprintf('\nPhase shifting analysis complete!\n');