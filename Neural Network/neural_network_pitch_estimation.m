function neural_network_pitch_estimation()
    % NEURAL NETWORK PITCH ESTIMATION
    % Creates and trains a deep neural network for pitch detection
    
    fprintf('Starting Neural Network Pitch Estimation...\n');
    
    % Step 1: Generate training data
    fprintf('Generating training data...\n');
    [X_train, y_train, X_val, y_val] = generate_training_data();
    
    % Step 2: Create and train the neural network
    fprintf('Creating neural network...\n');
    net = create_pitch_network();
    
    fprintf('Training neural network...\n');
    trained_net = train_network(net, X_train, y_train, X_val, y_val);
    
    % Step 3: Test on real audio
    fprintf('Testing on real audio file...\n');
    test_real_audio(trained_net);
    
    % Step 4: Compare with traditional methods
    fprintf('Comparing with traditional methods...\n');
    compare_methods(trained_net);
    
    fprintf('Neural network pitch estimation completed!\n');
end

%% Generate Training Data
function [X_train, y_train, X_val, y_val] = generate_training_data()
    % Parameters
    fs = 16000;           % Sample rate
    frame_length = 1024;  % Frame size
    n_mfcc = 13;         % Number of MFCC coefficients
    n_samples_per_freq = 200; % Samples per frequency
    
    % Frequency range for training (fundamental frequencies)
    frequencies = 80:5:800; % 80Hz to 800Hz in 5Hz steps
    
    X_all = [];
    y_all = [];
    
    fprintf('Generating synthetic data...\n');
    for i = 1:length(frequencies)
        freq = frequencies(i);
        if mod(i, 20) == 0
            fprintf('Processing frequency %d Hz (%d/%d)\n', freq, i, length(frequencies));
        end
        
        for j = 1:n_samples_per_freq
            % Generate complex harmonic signal
            signal = generate_complex_harmonic_signal(freq, fs, frame_length);
            
            % Extract features (MFCC + Spectral features)
            features = extract_features(signal, fs, n_mfcc);
            
            X_all = [X_all; features'];
            y_all = [y_all; freq];
        end
    end
    
    % Add noise variations
    fprintf('Adding noise variations...\n');
    X_noise = [];
    y_noise = [];
    noise_levels = [0.05, 0.1, 0.15, 0.2]; % Different noise levels
    
    for noise_level = noise_levels
        for i = 1:length(frequencies)
            freq = frequencies(i);
            for j = 1:50 % Fewer samples with noise
                signal = generate_complex_harmonic_signal(freq, fs, frame_length);
                % Add noise
                noise = noise_level * randn(size(signal));
                signal = signal + noise;
                
                features = extract_features(signal, fs, n_mfcc);
                X_noise = [X_noise; features'];
                y_noise = [y_noise; freq];
            end
        end
    end
    
    % Combine all data
    X_all = [X_all; X_noise];
    y_all = [y_all; y_noise];
    
    % Normalize features
    X_all = normalize(X_all, 'range');
    
    % Split into training and validation sets
    train_ratio = 0.8;
    n_train = round(train_ratio * length(y_all));
    
    % Shuffle data
    idx = randperm(length(y_all));
    X_shuffled = X_all(idx, :);
    y_shuffled = y_all(idx);
    
    X_train = X_shuffled(1:n_train, :);
    y_train = y_shuffled(1:n_train);
    X_val = X_shuffled(n_train+1:end, :);
    y_val = y_shuffled(n_train+1:end);
    
    fprintf('Training data: %d samples\n', length(y_train));
    fprintf('Validation data: %d samples\n', length(y_val));
end

%% Generate Complex Harmonic Signal
function signal = generate_complex_harmonic_signal(f0, fs, length_samples)
    t = (0:length_samples-1) / fs;
    signal = zeros(size(t));
    
    % Add fundamental and harmonics with realistic amplitudes
    harmonics = [1, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08]; % Amplitude decay
    
    for h = 1:min(length(harmonics), floor(fs/2/f0))
        freq = h * f0;
        if freq < fs/2 % Avoid aliasing
            amplitude = harmonics(h);
            % Add slight frequency modulation for realism
            phase_mod = 0.02 * sin(2*pi*5*t); % 5Hz vibrato
            signal = signal + amplitude * sin(2*pi*freq*t + phase_mod);
        end
    end
    
    % Apply envelope
    envelope = 0.5 * (1 + cos(2*pi*(0:length_samples-1)/length_samples - pi));
    signal = signal .* envelope;
end

%% Extract Features
function features = extract_features(signal, fs, n_mfcc)
    % Apply window
    windowed_signal = signal .* hamming(length(signal))';
    
    % MFCC features
    mfccs = mfcc(windowed_signal, fs, 'NumCoeffs', n_mfcc);
    
    % Spectral features
    fft_signal = fft(windowed_signal);
    magnitude = abs(fft_signal(1:floor(length(fft_signal)/2)));
    
    % Spectral centroid
    freqs = (0:length(magnitude)-1) * fs / (2*length(magnitude));
    spectral_centroid = sum(freqs .* magnitude') / sum(magnitude);
    
    % Spectral rolloff
    cumsum_mag = cumsum(magnitude);
    rolloff_threshold = 0.85 * cumsum_mag(end);
    rolloff_idx = find(cumsum_mag >= rolloff_threshold, 1);
    spectral_rolloff = freqs(rolloff_idx);
    
    % Zero crossing rate
    zcr = sum(abs(diff(sign(signal)))) / (2 * length(signal));
    
    % Spectral flux
    if length(magnitude) > 1
        spectral_flux = sqrt(mean(diff(magnitude).^2));
    else
        spectral_flux = 0;
    end
    
    % Combine all features
    features = [mfccs(:); spectral_centroid; spectral_rolloff; zcr; spectral_flux];
end

%% Create Neural Network
function net = create_pitch_network()
    % Define network architecture
    inputSize = 17; % 13 MFCC + 4 spectral features
    hiddenLayerSize1 = 128;
    hiddenLayerSize2 = 64;
    hiddenLayerSize3 = 32;
    outputSize = 1;
    
    % Create network layers
    layers = [
        featureInputLayer(inputSize, 'Name', 'input')
        fullyConnectedLayer(hiddenLayerSize1, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.3, 'Name', 'dropout1')
        
        fullyConnectedLayer(hiddenLayerSize2, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        dropoutLayer(0.3, 'Name', 'dropout2')
        
        fullyConnectedLayer(hiddenLayerSize3, 'Name', 'fc3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        dropoutLayer(0.2, 'Name', 'dropout3')
        
        fullyConnectedLayer(outputSize, 'Name', 'output')
        regressionLayer('Name', 'regression')
    ];
    
    net = layerGraph(layers);
end

%% Train Neural Network
function trained_net = train_network(net, X_train, y_train, X_val, y_val)
    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 20, ...
        'ValidationData', {X_val', y_val}, ...
        'ValidationFrequency', 50, ...
        'Verbose', true, ...
        'VerboseFrequency', 10, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto');
    
    % Train the network
    trained_net = trainNetwork(X_train', y_train, net, options);
end

%% Test on Real Audio
function test_real_audio(trained_net)
    try
        % Load audio file (replace with your file)
        [audio, fs] = audioread('your_audio_file.wav');
        
        if size(audio, 2) > 1
            audio = mean(audio, 2); % Convert to mono
        end
        
        % Resample if necessary
        if fs ~= 16000
            audio = resample(audio, 16000, fs);
            fs = 16000;
        end
        
        % Parameters
        frame_length = 1024;
        hop_length = 512;
        n_mfcc = 13;
        
        % Extract features and predict pitch
        pitches_nn = [];
        num_frames = floor((length(audio) - frame_length) / hop_length) + 1;
        
        fprintf('Processing %d frames...\n', num_frames);
        
        for i = 1:num_frames
            start_idx = (i-1) * hop_length + 1;
            end_idx = start_idx + frame_length - 1;
            
            if end_idx <= length(audio)
                frame = audio(start_idx:end_idx);
                features = extract_features(frame', fs, n_mfcc);
                
                % Normalize features (same as training)
                features_norm = normalize(features', 'range');
                
                % Predict pitch
                predicted_pitch = predict(trained_net, features_norm');
                pitches_nn = [pitches_nn; predicted_pitch];
            end
        end
        
        % Plot results
        time = (0:length(pitches_nn)-1) * hop_length / fs;
        
        figure('Position', [100, 100, 1000, 600]);
        subplot(2,1,1);
        plot((1:length(audio))/fs, audio, 'k-', 'LineWidth', 0.5);
        title('Original Audio Signal', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Time (s)'); ylabel('Amplitude');
        grid on;
        
        subplot(2,1,2);
        plot(time, pitches_nn, 'r-', 'LineWidth', 2);
        title('Neural Network Pitch Estimation', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Time (s)'); ylabel('Frequency (Hz)');
        ylim([50, 800]); grid on;
        
    catch ME
        fprintf('Error testing real audio: %s\n', ME.message);
        fprintf('Testing with synthetic signal instead...\n');
        test_synthetic_signal(trained_net);
    end
end

%% Test with Synthetic Signal
function test_synthetic_signal(trained_net)
    fs = 16000;
    duration = 3; % 3 seconds
    t = (0:1/fs:duration-1/fs);
    
    % Create a signal with varying pitch
    f0_base = 200; % Base frequency 200 Hz
    frequency_modulation = 50 * sin(2*pi*0.5*t); % ±50 Hz modulation at 0.5 Hz
    instantaneous_freq = f0_base + frequency_modulation;
    
    % Generate signal
    signal = sin(2*pi*cumsum(instantaneous_freq)/fs);
    
    % Add harmonics
    signal = signal + 0.5*sin(2*pi*2*cumsum(instantaneous_freq)/fs);
    signal = signal + 0.3*sin(2*pi*3*cumsum(instantaneous_freq)/fs);
    
    % Process with neural network
    frame_length = 1024;
    hop_length = 512;
    n_mfcc = 13;
    
    pitches_nn = [];
    num_frames = floor((length(signal) - frame_length) / hop_length) + 1;
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + frame_length - 1;
        
        if end_idx <= length(signal)
            frame = signal(start_idx:end_idx);
            features = extract_features(frame, fs, n_mfcc);
            features_norm = normalize(features', 'range');
            predicted_pitch = predict(trained_net, features_norm');
            pitches_nn = [pitches_nn; predicted_pitch];
        end
    end
    
    % Plot results
    time_signal = (1:length(signal))/fs;
    time_pitch = (0:length(pitches_nn)-1) * hop_length / fs;
    time_true = (0:length(instantaneous_freq)-1)/fs;
    
    figure('Position', [200, 200, 1000, 700]);
    subplot(3,1,1);
    plot(time_signal, signal, 'k-', 'LineWidth', 0.8);
    title('Synthetic Test Signal with Frequency Modulation', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    
    subplot(3,1,2);
    plot(time_true, instantaneous_freq, 'b-', 'LineWidth', 2);
    title('True Instantaneous Frequency', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Frequency (Hz)'); grid on;
    
    subplot(3,1,3);
    plot(time_pitch, pitches_nn, 'r-', 'LineWidth', 2);
    title('Neural Network Pitch Estimation', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Frequency (Hz)'); grid on;
    
    % Calculate error
    if length(pitches_nn) == length(instantaneous_freq)
        error = abs(pitches_nn' - instantaneous_freq);
        mean_error = mean(error);
        fprintf('Mean absolute error: %.2f Hz\n', mean_error);
    end
end

%% Compare with Traditional Methods
function compare_methods(trained_net)
    % Generate test signal
    fs = 16000;
    duration = 2;
    f0 = 220; % A3 note
    t = (0:1/fs:duration-1/fs);
    
    % Complex harmonic signal
    signal = sin(2*pi*f0*t) + 0.6*sin(2*pi*2*f0*t) + 0.4*sin(2*pi*3*f0*t);
    signal = signal + 0.1*randn(size(signal)); % Add noise
    
    % Parameters
    frame_length = 1024;
    hop_length = 512;
    n_mfcc = 13;
    
    % Neural Network Method
    pitches_nn = [];
    num_frames = floor((length(signal) - frame_length) / hop_length) + 1;
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + frame_length - 1;
        
        if end_idx <= length(signal)
            frame = signal(start_idx:end_idx);
            features = extract_features(frame, fs, n_mfcc);
            features_norm = normalize(features', 'range');
            predicted_pitch = predict(trained_net, features_norm');
            pitches_nn = [pitches_nn; predicted_pitch];
        end
    end
    
    % Traditional FFT Method
    pitches_fft = [];
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + frame_length - 1;
        
        if end_idx <= length(signal)
            frame = signal(start_idx:end_idx);
            windowed_frame = frame .* hamming(frame_length)';
            fft_frame = fft(windowed_frame);
            magnitude = abs(fft_frame(1:frame_length/2));
            freqs = (0:frame_length/2-1) * fs / frame_length;
            [~, max_idx] = max(magnitude);
            pitches_fft = [pitches_fft; freqs(max_idx)];
        end
    end
    
    % Plot comparison
    time = (0:length(pitches_nn)-1) * hop_length / fs;
    
    figure('Position', [300, 300, 1000, 600]);
    plot(time, pitches_fft, 'b-', 'LineWidth', 2, 'DisplayName', 'FFT Method');
    hold on;
    plot(time, pitches_nn, 'r-', 'LineWidth', 2, 'DisplayName', 'Neural Network');
    plot(time, f0*ones(size(time)), 'g--', 'LineWidth', 2, 'DisplayName', 'True Pitch');
    
    title('Comparison: Neural Network vs Traditional FFT', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    legend('show', 'Location', 'best');
    grid on; ylim([f0-50, f0+100]);
    
    % Calculate accuracy
    error_fft = abs(pitches_fft - f0);
    error_nn = abs(pitches_nn' - f0);
    
    fprintf('\n=== ACCURACY COMPARISON ===\n');
    fprintf('FFT Method - Mean Error: %.2f Hz, Std: %.2f Hz\n', mean(error_fft), std(error_fft));
    fprintf('Neural Network - Mean Error: %.2f Hz, Std: %.2f Hz\n', mean(error_nn), std(error_nn));
    
    if mean(error_nn) < mean(error_fft)
        fprintf('Neural Network is MORE accurate!\n');
    else
        fprintf('FFT method is more accurate for this test.\n');
    end
end