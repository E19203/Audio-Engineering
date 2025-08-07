%% NON-NEGATIVE MATRIX FACTORIZATION (NMF) FOR AUDIO STEM SEPARATION
% Advanced NMF implementation for separating audio sources

clc; clear; close all;

% INPUT YOUR AUDIO FILE HERE
input_file = 'Music.mp3';  % Change this to your audio file path
output_prefix = 'nmf_separated';   % Output files prefix

% NMF Parameters
num_sources = 4;           % Number of sources to separate (vocals, drums, bass, other)
num_iterations = 100;      % NMF iterations
window_size = 2048;        % STFT window size
hop_size = 512;           % STFT hop size

% Check if file exists
if ~exist(input_file, 'file')
    error('File not found: %s\nPlease check the file path.', input_file);
end

% Load audio file
fprintf('Loading audio file: %s\n', input_file);
[audio_data, fs] = audioread(input_file);
fprintf('Audio loaded: %.2f seconds, %d Hz sample rate\n', length(audio_data)/fs, fs);

% Convert to mono if stereo
if size(audio_data, 2) > 1
    audio_mono = mean(audio_data, 2);
    fprintf('Converted stereo to mono\n');
else
    audio_mono = audio_data;
end

% Compute Short-Time Fourier Transform (STFT)
fprintf('Computing STFT...\n');
[S, f, t] = spectrogram(audio_mono, hann(window_size), window_size - hop_size, window_size, fs);
V = abs(S);  % Magnitude spectrogram (non-negative)
phase_info = angle(S);  % Store phase for reconstruction

fprintf('Spectrogram size: %d x %d (freq x time)\n', size(V, 1), size(V, 2));

% Apply NMF for source separation
fprintf('Applying NMF with %d sources...\n', num_sources);
[W, H, cost_history] = nmf_multiplicative_update(V, num_sources, num_iterations);

% Reconstruct individual sources
fprintf('Reconstructing separated sources...\n');
separated_sources = cell(num_sources, 1);
original_length = length(audio_mono);

for src = 1:num_sources
    % Reconstruct magnitude spectrogram for this source
    V_reconstructed = W(:, src) * H(src, :);
    
    % Apply original phase information
    S_reconstructed = V_reconstructed .* exp(1i * phase_info);
    
    % Inverse STFT to get time-domain signal
    reconstructed_audio = istft_custom(S_reconstructed, window_size, hop_size);
    
    % Ensure same length as original audio
    if length(reconstructed_audio) > original_length
        % Truncate if too long
        reconstructed_audio = reconstructed_audio(1:original_length);
    elseif length(reconstructed_audio) < original_length
        % Pad with zeros if too short
        reconstructed_audio = [reconstructed_audio, zeros(1, original_length - length(reconstructed_audio))];
    end
    
    % Normalize and save
    source_normalized = reconstructed_audio / max(abs(reconstructed_audio) + eps) * 0.9;
    output_filename = sprintf('%s_source_%d.wav', output_prefix, src);
    audiowrite(output_filename, source_normalized', fs);
    fprintf('  Saved: %s\n', output_filename);
    
    separated_sources{src} = source_normalized;
end

% Visualization
visualize_nmf_results(V, W, H, f, t, cost_history, separated_sources, fs, audio_mono);

fprintf('\nNMF stem separation complete!\n');
fprintf('Separated %d sources saved with prefix: %s\n', num_sources, output_prefix);

%% NMF MULTIPLICATIVE UPDATE ALGORITHM
function [W, H, cost_history] = nmf_multiplicative_update(V, num_components, max_iterations)
    % NMF using multiplicative update rules
    % V ≈ W * H, where all matrices are non-negative
    
    [n_freq, n_time] = size(V);
    
    % Initialize W and H randomly (non-negative)
    W = rand(n_freq, num_components) + eps;
    H = rand(num_components, n_time) + eps;
    
    cost_history = zeros(max_iterations, 1);
    
    fprintf('NMF Progress: ');
    for iter = 1:max_iterations
        if mod(iter, 10) == 0
            fprintf('%d ', iter);
        end
        
        % Multiplicative update rules
        % Update H
        WTW = W' * W;
        WTV = W' * V;
        H = H .* (WTV ./ (WTW * H + eps));
        
        % Update W  
        HHT = H * H';
        VHT = V * H';
        W = W .* (VHT ./ (W * HHT + eps));
        
        % Normalize W columns to unit norm
        W_norms = sqrt(sum(W.^2, 1));
        W = W ./ (W_norms + eps);
        H = H .* (W_norms' + eps);
        
        % Calculate reconstruction error (Frobenius norm)
        reconstruction = W * H;
        cost_history(iter) = norm(V - reconstruction, 'fro')^2;
        
        % Early stopping if convergence achieved
        if iter > 10 && abs(cost_history(iter) - cost_history(iter-1)) / cost_history(iter-1) < 1e-6
            cost_history = cost_history(1:iter);
            fprintf('\nConverged at iteration %d\n', iter);
            break;
        end
    end
    
    if iter == max_iterations
        fprintf('\nReached maximum iterations\n');
    end
end

%% CUSTOM INVERSE STFT FUNCTION
function audio_reconstructed = istft_custom(S, window_size, hop_size)
    % Custom inverse STFT implementation
    
    [n_freq, n_frames] = size(S);
    audio_length = (n_frames - 1) * hop_size + window_size;
    audio_reconstructed = zeros(1, audio_length);
    window = hann(window_size);
    
    for frame = 1:n_frames
        % Compute IFFT
        if n_freq == window_size / 2 + 1  % One-sided spectrum
            spectrum = [S(:, frame); conj(flipud(S(2:end-1, frame)))];
        else
            spectrum = S(:, frame);
        end
        
        time_frame = real(ifft(spectrum, window_size));
        
        % Overlap and add
        start_idx = (frame - 1) * hop_size + 1;
        end_idx = start_idx + window_size - 1;
        
        if end_idx <= audio_length
            audio_reconstructed(start_idx:end_idx) = ...
                audio_reconstructed(start_idx:end_idx) + time_frame' .* window';
        end
    end
end

%% ADVANCED NMF WITH SOURCE PRIORS
function [W, H] = nmf_with_source_priors(V, num_components, max_iterations)
    % Enhanced NMF with source-specific constraints
    
    [n_freq, n_time] = size(V);
    
    % Initialize with source-informed priors
    W = initialize_source_templates(n_freq, num_components);
    H = rand(num_components, n_time) + eps;
    
    % Source-specific update parameters
    alpha = ones(num_components, 1);  % Sparsity parameters
    beta = ones(num_components, 1);   % Smoothness parameters
    
    for iter = 1:max_iterations
        % Standard multiplicative updates with regularization
        WTW = W' * W;
        WTV = W' * V;
        
        % Add temporal smoothness regularization to H
        H_smooth = [H(:, 1), H(:, 1:end-1)] + [H(:, 2:end), H(:, end)];
        H = H .* (WTV ./ (WTW * H + beta .* H_smooth + eps));
        
        % Update W with sparsity constraint
        HHT = H * H';
        VHT = V * H';
        W = W .* (VHT ./ (W * HHT + alpha .* W + eps));
        
        % Normalize
        W_norms = sqrt(sum(W.^2, 1));
        W = W ./ (W_norms + eps);
        H = H .* (W_norms' + eps);
    end
end

function W = initialize_source_templates(n_freq, num_components)
    % Initialize W with source-informed templates
    
    W = rand(n_freq, num_components) + eps;
    
    % Create frequency templates for different sources
    if num_components >= 4
        % Vocals: emphasis on mid frequencies
        vocal_template = exp(-((1:n_freq)' - n_freq/3).^2 / (n_freq/10)^2);
        W(:, 1) = vocal_template / max(vocal_template) + 0.1 * rand(n_freq, 1);
        
        % Drums: emphasis on low and high frequencies
        drum_template = exp(-((1:n_freq)' - n_freq/10).^2 / (n_freq/20)^2) + ...
                       exp(-((1:n_freq)' - 3*n_freq/4).^2 / (n_freq/8)^2);
        W(:, 2) = drum_template / max(drum_template) + 0.1 * rand(n_freq, 1);
        
        % Bass: emphasis on very low frequencies  
        bass_template = exp(-((1:n_freq)' - n_freq/20).^2 / (n_freq/40)^2);
        W(:, 3) = bass_template / max(bass_template) + 0.1 * rand(n_freq, 1);
        
        % Other: remaining components random
        for i = 4:num_components
            W(:, i) = rand(n_freq, 1) + eps;
        end
    end
end

%% VISUALIZATION FUNCTION
function visualize_nmf_results(V, W, H, f, t, cost_history, separated_sources, fs, original_audio)
    % Comprehensive visualization of NMF results
    
    figure('Name', 'NMF Audio Separation Results', 'Position', [100 50 1200 800]);
    
    % Original spectrogram
    subplot(3, 4, [1, 2]);
    imagesc(t, f/1000, 20*log10(V + eps));
    axis xy; colorbar;
    title('Original Audio Spectrogram');
    xlabel('Time (s)'); ylabel('Frequency (kHz)');
    
    % NMF basis functions (W)
    subplot(3, 4, 3);
    imagesc(1:size(W, 2), f/1000, W);
    axis xy; colorbar;
    title('NMF Basis Functions (W)');
    xlabel('Component'); ylabel('Frequency (kHz)');
    
    % NMF activation matrix (H)
    subplot(3, 4, 4);
    imagesc(t, 1:size(H, 1), H);
    axis xy; colorbar;
    title('NMF Activations (H)');
    xlabel('Time (s)'); ylabel('Component');
    
    % Cost function convergence
    subplot(3, 4, 5);
    semilogy(cost_history);
    title('NMF Convergence');
    xlabel('Iteration'); ylabel('Reconstruction Error');
    grid on;
    
    % Individual source spectrograms
    for src = 1:min(3, length(separated_sources))
        subplot(3, 4, 5 + src);
        [S_src, ~, ~] = spectrogram(separated_sources{src}, hann(2048), 1536, 2048, fs);
        imagesc(t, f/1000, 20*log10(abs(S_src) + eps));
        axis xy; colorbar;
        title(sprintf('Separated Source %d', src));
        xlabel('Time (s)'); ylabel('Frequency (kHz)');
    end
    
    % Time-domain comparison
    subplot(3, 4, [9, 10, 11, 12]);
    samples_to_show = min(5*fs, length(original_audio));
    time_vec = (0:samples_to_show-1) / fs;
    
    plot(time_vec, original_audio(1:samples_to_show), 'k', 'LineWidth', 1.5);
    hold on;
    
    colors = {'r', 'g', 'b', 'm'};
    for src = 1:min(length(separated_sources), 4)
        plot(time_vec, separated_sources{src}(1:samples_to_show), colors{src}, 'LineWidth', 1);
    end
    
    legend(['Original', arrayfun(@(x) sprintf('Source %d', x), 1:min(length(separated_sources), 4), 'UniformOutput', false)]);
    title('Time Domain Comparison (First 5 seconds)');
    xlabel('Time (s)'); ylabel('Amplitude');
    grid on;
end