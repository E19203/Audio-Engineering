function pitch_estimation_analysis()
    % PITCH ESTIMATION ANALYSIS
    % Implements multiple accurate pitch detection algorithms
    
    % Load audio file
    [audio, fs] = audioread('Music.mp3'); % Replace with your file
    
    % Convert to mono if stereo
    if size(audio, 2) > 1
        audio = mean(audio, 2);
    end
    
    % Parameters
    frame_length = 2048;
    hop_length = 512;
    
    % Apply all pitch detection methods
    fprintf('Processing audio file...\n');
    
    % Method 1: FFT-based pitch detection
    pitches_fft = fft_pitch_detection(audio, fs, frame_length, hop_length);
    
    % Method 2: Autocorrelation pitch detection
    pitches_autocorr = autocorr_pitch_detection(audio, fs, frame_length, hop_length);
    
    % Method 3: Harmonic Product Spectrum
    pitches_hps = hps_pitch_detection(audio, fs, frame_length, hop_length);
    
    % Method 4: YIN algorithm (most accurate)
    pitches_yin = yin_pitch_detection(audio, fs, frame_length, hop_length);
    
    % Method 5: Cepstral analysis
    pitches_cepstral = cepstral_pitch_detection(audio, fs, frame_length, hop_length);
    
    % Plot results
    plot_pitch_results(pitches_fft, pitches_autocorr, pitches_hps, pitches_yin, pitches_cepstral, fs, hop_length);
    
    % Display statistics
    display_statistics(pitches_fft, pitches_autocorr, pitches_hps, pitches_yin, pitches_cepstral);
end

%% Method 1: FFT-based Pitch Detection
function pitches = fft_pitch_detection(audio, fs, frame_length, hop_length)
    num_frames = floor((length(audio) - frame_length) / hop_length) + 1;
    pitches = zeros(num_frames, 1);
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + frame_length - 1;
        frame = audio(start_idx:end_idx);
        
        % Apply window
        frame = frame .* hamming(frame_length);
        
        % FFT
        fft_frame = fft(frame, frame_length);
        magnitude = abs(fft_frame(1:floor(frame_length/2)));
        
        % Find peak
        [~, max_idx] = max(magnitude);
        pitches(i) = (max_idx - 1) * fs / frame_length;
    end
end

%% Method 2: Autocorrelation Pitch Detection
function pitches = autocorr_pitch_detection(audio, fs, frame_length, hop_length)
    num_frames = floor((length(audio) - frame_length) / hop_length) + 1;
    pitches = zeros(num_frames, 1);
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + frame_length - 1;
        frame = audio(start_idx:end_idx);
        
        % Apply window
        frame = frame .* hamming(frame_length);
        
        % Autocorrelation
        autocorr_result = xcorr(frame, frame);
        autocorr_result = autocorr_result(frame_length:end);
        
        % Find first peak after zero lag
        min_period = round(fs / 800); % Minimum period for 800 Hz
        max_period = round(fs / 50);  % Maximum period for 50 Hz
        
        if max_period <= length(autocorr_result)
            [~, peak_idx] = max(autocorr_result(min_period:max_period));
            period = peak_idx + min_period - 1;
            pitches(i) = fs / period;
        else
            pitches(i) = 0;
        end
    end
end

%% Method 3: Harmonic Product Spectrum
function pitches = hps_pitch_detection(audio, fs, frame_length, hop_length)
    num_frames = floor((length(audio) - frame_length) / hop_length) + 1;
    pitches = zeros(num_frames, 1);
    num_harmonics = 5;
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + frame_length - 1;
        frame = audio(start_idx:end_idx);
        
        % Apply window
        frame = frame .* hamming(frame_length);
        
        % FFT
        fft_frame = fft(frame, frame_length);
        magnitude = abs(fft_frame(1:floor(frame_length/2)));
        
        % Harmonic Product Spectrum
        hps = magnitude;
        for h = 2:num_harmonics
            downsampled = magnitude(1:h:end);
            hps_length = min(length(hps), length(downsampled));
            hps(1:hps_length) = hps(1:hps_length) .* downsampled(1:hps_length);
        end
        
        % Find peak
        [~, max_idx] = max(hps);
        pitches(i) = (max_idx - 1) * fs / frame_length;
    end
end

%% Method 4: YIN Algorithm (Most Accurate)
function pitches = yin_pitch_detection(audio, fs, frame_length, hop_length)
    num_frames = floor((length(audio) - frame_length) / hop_length) + 1;
    pitches = zeros(num_frames, 1);
    threshold = 0.1;
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + frame_length - 1;
        frame = audio(start_idx:end_idx);
        
        % Difference function
        df = zeros(frame_length, 1);
        for tau = 1:frame_length
            if tau == 1
                df(tau) = 1;
            else
                for j = 1:(frame_length - tau)
                    df(tau) = df(tau) + (frame(j) - frame(j + tau - 1))^2;
                end
            end
        end
        
        % Cumulative mean normalized difference
        cmndf = zeros(frame_length, 1);
        cmndf(1) = 1;
        for tau = 2:frame_length
            cmndf(tau) = df(tau) / (sum(df(2:tau)) / (tau - 1));
        end
        
        % Find minimum below threshold
        min_tau = 0;
        for tau = 2:frame_length
            if cmndf(tau) < threshold
                min_tau = tau;
                break;
            end
        end
        
        if min_tau > 0
            pitches(i) = fs / min_tau;
        else
            pitches(i) = 0;
        end
    end
end

%% Method 5: Cepstral Analysis
function pitches = cepstral_pitch_detection(audio, fs, frame_length, hop_length)
    num_frames = floor((length(audio) - frame_length) / hop_length) + 1;
    pitches = zeros(num_frames, 1);
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + frame_length - 1;
        frame = audio(start_idx:end_idx);
        
        % Apply window
        frame = frame .* hamming(frame_length);
        
        % Cepstral analysis
        fft_frame = fft(frame, frame_length);
        log_magnitude = log(abs(fft_frame) + eps);
        cepstrum = real(ifft(log_magnitude));
        
        % Find peak in cepstrum
        min_period = round(fs / 800);
        max_period = round(fs / 50);
        
        if max_period <= length(cepstrum)
            [~, peak_idx] = max(cepstrum(min_period:max_period));
            period = peak_idx + min_period - 1;
            pitches(i) = fs / period;
        else
            pitches(i) = 0;
        end
    end
end

%% Plotting Function - Separate Figures
function plot_pitch_results(pitches_fft, pitches_autocorr, pitches_hps, pitches_yin, pitches_cepstral, fs, hop_length)
    time = (0:length(pitches_fft)-1) * hop_length / fs;
    
    % Figure 1: FFT-based Pitch Detection
    figure('Position', [100, 100, 800, 500]);
    plot(time, pitches_fft, 'b-', 'LineWidth', 2);
    title('FFT-based Pitch Detection', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)', 'FontSize', 12); 
    ylabel('Frequency (Hz)', 'FontSize', 12);
    grid on; ylim([0, 1000]);
    set(gca, 'FontSize', 11);
    
    % Figure 2: Autocorrelation Pitch Detection
    figure('Position', [150, 150, 800, 500]);
    plot(time, pitches_autocorr, 'r-', 'LineWidth', 2);
    title('Autocorrelation Pitch Detection', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)', 'FontSize', 12); 
    ylabel('Frequency (Hz)', 'FontSize', 12);
    grid on; ylim([0, 1000]);
    set(gca, 'FontSize', 11);
    
    % Figure 3: Harmonic Product Spectrum
    figure('Position', [200, 200, 800, 500]);
    plot(time, pitches_hps, 'g-', 'LineWidth', 2);
    title('Harmonic Product Spectrum', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)', 'FontSize', 12); 
    ylabel('Frequency (Hz)', 'FontSize', 12);
    grid on; ylim([0, 1000]);
    set(gca, 'FontSize', 11);
    
    % Figure 4: YIN Algorithm (Most Accurate)
    figure('Position', [250, 250, 800, 500]);
    plot(time, pitches_yin, 'm-', 'LineWidth', 2);
    title('YIN Algorithm (Most Accurate)', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)', 'FontSize', 12); 
    ylabel('Frequency (Hz)', 'FontSize', 12);
    grid on; ylim([0, 1000]);
    set(gca, 'FontSize', 11);
    
    % Figure 5: Cepstral Analysis
    figure('Position', [300, 300, 800, 500]);
    plot(time, pitches_cepstral, 'c-', 'LineWidth', 2);
    title('Cepstral Analysis', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)', 'FontSize', 12); 
    ylabel('Frequency (Hz)', 'FontSize', 12);
    grid on; ylim([0, 1000]);
    set(gca, 'FontSize', 11);
    
    % Figure 6: Comparison of All Methods
    figure('Position', [350, 350, 1000, 600]);
    plot(time, pitches_fft, 'b-', 'LineWidth', 1.5, 'DisplayName', 'FFT');
    hold on;
    plot(time, pitches_autocorr, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Autocorrelation');
    plot(time, pitches_hps, 'g-', 'LineWidth', 1.5, 'DisplayName', 'HPS');
    plot(time, pitches_yin, 'm-', 'LineWidth', 2, 'DisplayName', 'YIN');
    plot(time, pitches_cepstral, 'c-', 'LineWidth', 1.5, 'DisplayName', 'Cepstral');
    title('Comparison of All Pitch Detection Methods', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)', 'FontSize', 12); 
    ylabel('Frequency (Hz)', 'FontSize', 12);
    legend('show', 'Location', 'best', 'FontSize', 10); 
    grid on; ylim([0, 1000]);
    set(gca, 'FontSize', 11);
    hold off;
end

%% Statistics Display
function display_statistics(pitches_fft, pitches_autocorr, pitches_hps, pitches_yin, pitches_cepstral)
    fprintf('\n=== PITCH DETECTION STATISTICS ===\n');
    
    methods = {'FFT', 'Autocorr', 'HPS', 'YIN', 'Cepstral'};
    pitches_all = {pitches_fft, pitches_autocorr, pitches_hps, pitches_yin, pitches_cepstral};
    
    for i = 1:length(methods)
        pitches = pitches_all{i};
        valid_pitches = pitches(pitches > 0);
        
        fprintf('%s Method:\n', methods{i});
        fprintf('  Mean Pitch: %.2f Hz\n', mean(valid_pitches));
        fprintf('  Median Pitch: %.2f Hz\n', median(valid_pitches));
        fprintf('  Std Deviation: %.2f Hz\n', std(valid_pitches));
        fprintf('  Valid Frames: %d/%d (%.1f%%)\n', length(valid_pitches), length(pitches), 100*length(valid_pitches)/length(pitches));
        fprintf('\n');
    end
end