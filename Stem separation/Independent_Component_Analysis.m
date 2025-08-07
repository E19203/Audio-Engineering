%% MP3 STEM SEPARATION USING ICA

% INPUT YOUR MP3 FILE HERE
input_file = 'Music.mp3';  % Change this to your MP3 file path
output_prefix = 'separated';   % Output files will be named: separated_source_1.wav, etc.

% Check if file exists
if ~exist(input_file, 'file')
    error('File not found: %s\nPlease check the file path.', input_file);
end

% Load MP3 file
fprintf('Loading MP3 file: %s\n', input_file);
[audio_data, fs] = audioread(input_file);
fprintf('Audio loaded: %.2f seconds, %d Hz sample rate\n', length(audio_data)/fs, fs);

% Prepare mixed signals for ICA
if size(audio_data, 2) == 1
    % Mono file - create artificial mixing for demonstration
    fprintf('Mono file detected. Creating stereo mix for separation.\n');
    delay_samples = round(0.02 * fs); % 20ms delay
    mixed1 = audio_data;
    mixed2 = 0.7 * [zeros(delay_samples,1); audio_data(1:end-delay_samples)] + ...
             0.3 * audio_data;
    X = [mixed1'; mixed2'];
else
    % Stereo file - use left/right channels
    fprintf('Stereo file detected. Using L/R channels for separation.\n');
    X = audio_data';
end

% Apply ICA separation
fprintf('Applying ICA separation...\n');
num_sources = size(X, 1);
[S_separated, W] = fastICA_separation(X, num_sources);

% Normalize and save separated sources
fprintf('Saving separated sources:\n');
for i = 1:num_sources
    % Normalize to prevent clipping
    source_normalized = S_separated(i,:) / max(abs(S_separated(i,:))) * 0.9;
    
    % Save as WAV file
    output_filename = sprintf('%s_source_%d.wav', output_prefix, i);
    audiowrite(output_filename, source_normalized', fs);
    fprintf('  Saved: %s\n', output_filename);
end

% Plot results
figure('Name', 'MP3 Stem Separation Results', 'Position', [100 100 800 600]);

% Show first 5 seconds of audio
samples_to_show = min(5*fs, length(audio_data));
time_vector = (0:samples_to_show-1) / fs;

subplot(2+num_sources, 1, 1);
plot(time_vector, audio_data(1:samples_to_show, 1));
title('Original Audio (Left Channel)');
xlabel('Time (s)'); ylabel('Amplitude');

if size(audio_data, 2) == 2
    subplot(2+num_sources, 1, 2);
    plot(time_vector, audio_data(1:samples_to_show, 2));
    title('Original Audio (Right Channel)');
    xlabel('Time (s)'); ylabel('Amplitude');
end

for i = 1:num_sources
    subplot(2+num_sources, 1, 2+i);
    plot(time_vector, S_separated(i, 1:samples_to_show));
    title(sprintf('Separated Source %d', i));
    xlabel('Time (s)'); ylabel('Amplitude');
end

fprintf('\nStem separation complete!\n');
fprintf('Separated sources saved as WAV files with prefix: %s\n', output_prefix);

%% FAST ICA FUNCTION
function [S, W] = fastICA_separation(X, num_components)
    % Fast ICA algorithm for source separation
    
    % Center the data
    X_mean = mean(X, 2);
    X_centered = X - X_mean;
    
    % Whiten the data
    cov_matrix = cov(X_centered');
    [E, D] = eig(cov_matrix);
    whitening_matrix = E * diag(1./sqrt(diag(D))) * E';
    X_white = whitening_matrix * X_centered;
    
    % Initialize
    [n_channels, n_samples] = size(X_white);
    W = zeros(num_components, n_channels);
    
    for comp = 1:num_components
        % Random initialization
        w = randn(n_channels, 1);
        w = w / norm(w);
        
        % Fixed-point iteration
        for iter = 1:200
            w_old = w;
            
            % Compute output
            y = w' * X_white;
            
            % Nonlinearity: tanh
            g = tanh(y);
            g_prime = 1 - g.^2;
            
            % Update rule
            w_new = mean(X_white .* g, 2) - mean(g_prime) * w;
            
            % Orthogonalize against previous components
            for prev = 1:comp-1
                w_new = w_new - (w_new' * W(prev,:)') * W(prev,:)';
            end
            
            % Normalize
            w = w_new / norm(w_new);
            
            % Check convergence
            if abs(abs(w' * w_old) - 1) < 1e-6
                break;
            end
        end
        
        W(comp, :) = w';
    end
    
    % Extract sources
    S = W * X_white;
    
    % Add back mean
    S = S + W * whitening_matrix * X_mean;
end