import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def generate_training_data():
    """Generate synthetic training data for pitch estimation"""
    print("Generating training data...")
    
    # Parameters
    fs = 16000
    duration = 0.5  # 500ms per sample
    n_mfcc = 13
    n_samples_per_freq = 150
    
    # Frequency range (fundamental frequencies)
    frequencies = np.arange(80, 801, 5)  # 80Hz to 800Hz in 5Hz steps
    
    X_all = []
    y_all = []
    
    # Generate clean synthetic data
    for i, freq in enumerate(frequencies):
        if i % 20 == 0:
            print(f"Processing frequency {freq} Hz ({i}/{len(frequencies)})")
        
        for j in range(n_samples_per_freq):
            # Generate complex harmonic signal
            signal = generate_harmonic_signal(freq, fs, duration)
            
            # Extract features
            features = extract_features(signal, fs, n_mfcc)
            
            X_all.append(features)
            y_all.append(freq)
    
    # Add noisy variations
    print("Adding noise variations...")
    noise_levels = [0.05, 0.1, 0.15, 0.2]
    
    for noise_level in noise_levels:
        for i, freq in enumerate(frequencies[::2]):  # Every other frequency for noise
            for j in range(30):  # Fewer noisy samples
                signal = generate_harmonic_signal(freq, fs, duration)
                # Add noise
                signal += noise_level * np.random.randn(len(signal))
                
                features = extract_features(signal, fs, n_mfcc)
                X_all.append(features)
                y_all.append(freq)
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    print(f"Total samples generated: {len(y_all)}")
    return X_all, y_all

def generate_harmonic_signal(f0, fs, duration):
    """Generate a complex harmonic signal"""
    t = np.linspace(0, duration, int(fs * duration), False)
    signal = np.zeros(len(t))
    
    # Add fundamental and harmonics with realistic decay
    harmonics = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08]
    
    for h, amplitude in enumerate(harmonics, 1):
        freq = h * f0
        if freq < fs / 2:  # Avoid aliasing
            # Add slight frequency modulation for realism
            vibrato = 0.02 * np.sin(2 * np.pi * 5 * t)  # 5Hz vibrato
            signal += amplitude * np.sin(2 * np.pi * freq * t + vibrato)
    
    # Apply envelope
    envelope = 0.5 * (1 + np.cos(2 * np.pi * np.arange(len(t)) / len(t) - np.pi))
    signal *= envelope
    
    return signal

def extract_features(signal, fs, n_mfcc=13):
    """Extract features from audio signal"""
    # MFCC features
    mfccs = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=fs)
    spectral_centroid = np.mean(spectral_centroids)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=fs)
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
    zcr_mean = np.mean(zero_crossing_rate)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=fs)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    
    # Combine all features
    features = np.concatenate([
        mfccs_mean,
        [spectral_centroid, spectral_rolloff_mean, zcr_mean, spectral_bandwidth_mean]
    ])
    
    return features

def create_neural_network(input_size):
    """Create deep neural network for pitch estimation"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_size,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(1, activation='linear')  # Output layer for regression
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_network(model, X_train, y_train, X_val, y_val):
    """Train the neural network"""
    print("Training neural network...")
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_on_real_audio(model, scaler, audio_file="Music.mp3"):
    """Test the model on real audio"""
    try:
        if audio_file is None:
            # Generate test signal if no file provided
            print("No audio file provided. Testing on synthetic signal...")
            test_synthetic_signal(model, scaler)
            return
        
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Parameters
        frame_length = 2048
        hop_length = 512
        n_mfcc = 13
        
        # Process audio in frames
        pitches = []
        num_frames = (len(audio) - frame_length) // hop_length + 1
        
        print(f"Processing {num_frames} frames...")
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            
            if end <= len(audio):
                frame = audio[start:end]
                features = extract_features(frame, sr, n_mfcc)
                features_scaled = scaler.transform([features])
                
                pitch = model.predict(features_scaled, verbose=0)[0][0]
                pitches.append(pitch)
        
        # Plot results
        time = np.arange(len(pitches)) * hop_length / sr
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(audio)) / sr, audio, 'k-', alpha=0.7)
        plt.title('Original Audio Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(time, pitches, 'r-', linewidth=2)
        plt.title('Neural Network Pitch Estimation')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim([50, 800])
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Mean pitch: {np.mean(pitches):.2f} Hz")
        print(f"Pitch range: {np.min(pitches):.2f} - {np.max(pitches):.2f} Hz")
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        print("Testing on synthetic signal instead...")
        test_synthetic_signal(model, scaler)

def test_synthetic_signal(model, scaler):
    """Test on synthetic signal with known pitch"""
    print("Testing on synthetic signal...")
    
    # Generate test signal with frequency modulation
    fs = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(fs * duration), False)
    
    # Create frequency modulated signal
    f0_base = 220  # A3 note
    frequency_modulation = 50 * np.sin(2 * np.pi * 0.5 * t)  # ±50 Hz at 0.5 Hz
    instantaneous_freq = f0_base + frequency_modulation
    
    # Generate signal with harmonics
    signal = np.sin(2 * np.pi * np.cumsum(instantaneous_freq) / fs)
    signal += 0.5 * np.sin(2 * np.pi * 2 * np.cumsum(instantaneous_freq) / fs)
    signal += 0.3 * np.sin(2 * np.pi * 3 * np.cumsum(instantaneous_freq) / fs)
    
    # Process with neural network
    frame_length = 2048
    hop_length = 512
    n_mfcc = 13
    
    pitches = []
    num_frames = (len(signal) - frame_length) // hop_length + 1
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        
        if end <= len(signal):
            frame = signal[start:end]
            features = extract_features(frame, fs, n_mfcc)
            features_scaled = scaler.transform([features])
            
            pitch = model.predict(features_scaled, verbose=0)[0][0]
            pitches.append(pitch)
    
    # Plot results
    time_signal = np.arange(len(signal)) / fs
    time_pitch = np.arange(len(pitches)) * hop_length / fs
    time_true = np.arange(len(instantaneous_freq)) / fs
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_signal, signal, 'k-', alpha=0.7)
    plt.title('Synthetic Test Signal with Frequency Modulation')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_true, instantaneous_freq, 'b-', linewidth=2, label='True Pitch')
    plt.title('True Instantaneous Frequency')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(time_pitch, pitches, 'r-', linewidth=2, label='Neural Network')
    plt.title('Neural Network Pitch Estimation')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate error if lengths match
    if len(pitches) == len(instantaneous_freq):
        error = np.abs(np.array(pitches) - instantaneous_freq)
        print(f"Mean absolute error: {np.mean(error):.2f} Hz")
        print(f"Max error: {np.max(error):.2f} Hz")
    else:
        print(f"Length mismatch: pitches={len(pitches)}, true={len(instantaneous_freq)}")

def compare_methods(model, scaler):
    """Compare neural network with traditional methods"""
    print("Comparing methods...")
    
    # Generate test signal
    fs = 16000
    duration = 2.0
    f0 = 220  # A3 note
    t = np.linspace(0, duration, int(fs * duration), False)
    
    # Complex harmonic signal with noise
    signal = np.sin(2 * np.pi * f0 * t)
    signal += 0.6 * np.sin(2 * np.pi * 2 * f0 * t)
    signal += 0.4 * np.sin(2 * np.pi * 3 * f0 * t)
    signal += 0.1 * np.random.randn(len(signal))  # Add noise
    
    # Neural Network Method
    frame_length = 2048
    hop_length = 512
    n_mfcc = 13
    
    pitches_nn = []
    num_frames = (len(signal) - frame_length) // hop_length + 1
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        
        if end <= len(signal):
            frame = signal[start:end]
            features = extract_features(frame, fs, n_mfcc)
            features_scaled = scaler.transform([features])
            
            pitch = model.predict(features_scaled, verbose=0)[0][0]
            pitches_nn.append(pitch)
    
    # Traditional FFT Method
    pitches_fft = []
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        
        if end <= len(signal):
            frame = signal[start:end]
            windowed_frame = frame * np.hanning(len(frame))
            
            fft_frame = np.fft.fft(windowed_frame)
            magnitude = np.abs(fft_frame[:len(fft_frame)//2])
            freqs = np.fft.fftfreq(len(fft_frame), 1/fs)[:len(fft_frame)//2]
            
            # Find peak frequency
            peak_idx = np.argmax(magnitude)
            pitches_fft.append(freqs[peak_idx])
    
    # Plot comparison
    time = np.arange(len(pitches_nn)) * hop_length / fs
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, pitches_fft, 'b-', linewidth=2, label='FFT Method')
    plt.plot(time, pitches_nn, 'r-', linewidth=2, label='Neural Network')
    plt.axhline(y=f0, color='g', linestyle='--', linewidth=2, label='True Pitch')
    
    plt.title('Comparison: Neural Network vs Traditional FFT')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.ylim([f0-50, f0+100])
    
    plt.show()
    
    # Calculate accuracy
    error_fft = np.abs(np.array(pitches_fft) - f0)
    error_nn = np.abs(np.array(pitches_nn) - f0)
    
    print("\n=== ACCURACY COMPARISON ===")
    print(f"FFT Method - Mean Error: {np.mean(error_fft):.2f} Hz, Std: {np.std(error_fft):.2f} Hz")
    print(f"Neural Network - Mean Error: {np.mean(error_nn):.2f} Hz, Std: {np.std(error_nn):.2f} Hz")
    
    if np.mean(error_nn) < np.mean(error_fft):
        print("Neural Network is MORE accurate!")
    else:
        print("FFT method is more accurate for this test.")

def main():
    """Main function"""
    print("=== NEURAL NETWORK PITCH ESTIMATION ===")
    
    # Generate training data
    X, y = generate_training_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {len(X_train_scaled)}")
    print(f"Validation samples: {len(X_val_scaled)}")
    print(f"Test samples: {len(X_test_scaled)}")
    print(f"Feature dimensions: {X_train_scaled.shape[1]}")
    
    # Create and train model
    model = create_neural_network(X_train_scaled.shape[1])
    model.summary()
    
    trained_model, history = train_network(model, X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_loss, test_mae = trained_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.2f} Hz")
    
    # Test on real audio (replace with your audio file)
    print("\n=== TESTING ON AUDIO ===")
    test_on_real_audio(trained_model, scaler, audio_file="Music.mp3")  # Set to your audio file path
    
    # Compare methods
    print("\n=== METHOD COMPARISON ===")
    compare_methods(trained_model, scaler)
    
    print("\nNeural network pitch estimation completed!")

if __name__ == "__main__":
    main()
