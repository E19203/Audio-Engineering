import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import spectrogram, stft, istft
from scipy import signal
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import audio libraries with fallbacks
try:
    import librosa
    # Test if librosa.display works with current matplotlib version
    try:
        import librosa.display
        LIBROSA_DISPLAY_AVAILABLE = True
    except ImportError:
        LIBROSA_DISPLAY_AVAILABLE = False
        print("Warning: librosa.display not compatible with current matplotlib version, using scipy")
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    LIBROSA_DISPLAY_AVAILABLE = False
    print("Warning: librosa not available")

# Try alternative audio loading libraries
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

if not LIBROSA_AVAILABLE and not SOUNDFILE_AVAILABLE and not PYDUB_AVAILABLE:
    print("Warning: No audio libraries available. Install one of: librosa, soundfile, or pydub")

class AudioSubspaceFilter:
    def __init__(self, n_components=10):
        """
        Initialize the Audio Subspace Filter with PCA
        
        Args:
            n_components (int): Number of principal components to retain
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.original_signal = None
        self.processed_signal = None
        self.sample_rate = None
        
    def load_audio_file(self, audio_file, duration=None, sr=22050):
        """
        Load audio file using multiple fallback methods
        
        Args:
            audio_file (str): Path to audio file
            duration (float): Duration to load (None for full file)
            sr (int): Target sample rate
        
        Returns:
            tuple: (audio_data, sample_rate)
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        print(f"Loading audio file: {audio_file}")
        
        # Method 1: Try librosa (best option)
        if LIBROSA_AVAILABLE:
            try:
                if duration:
                    audio_data, sample_rate = librosa.load(audio_file, duration=duration, sr=sr)
                else:
                    audio_data, sample_rate = librosa.load(audio_file, sr=sr)
                print(f"Loaded with librosa - Shape: {audio_data.shape}, SR: {sample_rate}")
                return audio_data, sample_rate
            except Exception as e:
                print(f"Librosa failed: {e}, trying next method...")
        
        # Method 2: Try soundfile
        if SOUNDFILE_AVAILABLE:
            try:
                audio_data, sample_rate = sf.read(audio_file)
                # Convert stereo to mono if needed
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                # Resample if needed (basic method)
                if sample_rate != sr:
                    # Simple resampling
                    ratio = sr / sample_rate
                    new_length = int(len(audio_data) * ratio)
                    audio_data = np.interp(np.linspace(0, len(audio_data), new_length), 
                                         np.arange(len(audio_data)), audio_data)
                    sample_rate = sr
                # Truncate if duration specified
                if duration:
                    max_samples = int(duration * sample_rate)
                    audio_data = audio_data[:max_samples]
                print(f"Loaded with soundfile - Shape: {audio_data.shape}, SR: {sample_rate}")
                return audio_data, sample_rate
            except Exception as e:
                print(f"Soundfile failed: {e}, trying next method...")
        
        # Method 3: Try pydub (works well with MP3)
        if PYDUB_AVAILABLE:
            try:
                # Load with pydub
                audio = AudioSegment.from_file(audio_file)
                
                # Convert to mono
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # Set sample rate
                audio = audio.set_frame_rate(sr)
                
                # Truncate if duration specified
                if duration:
                    audio = audio[:int(duration * 1000)]  # pydub uses milliseconds
                
                # Convert to numpy array
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                # Normalize
                audio_data = audio_data / np.max(np.abs(audio_data))
                
                print(f"Loaded with pydub - Shape: {audio_data.shape}, SR: {sr}")
                return audio_data, sr
            except Exception as e:
                print(f"Pydub failed: {e}")
        
        raise Exception("Could not load audio file with any available method. "
                       "Please install librosa, soundfile, or pydub.")
    
    def phase_01_import_signal(self, audio_file=None, duration=10, sr=22050):
        """
        Phase 01: Import the audio signal data
        
        Args:
            audio_file (str): Path to audio file (optional)
            duration (float): Duration of signal to load
            sr (int): Sample rate
        """
        if audio_file:
            try:
                self.original_signal, self.sample_rate = self.load_audio_file(
                    audio_file, duration=duration, sr=sr
                )
                print(f"Successfully loaded: {audio_file}")
            except Exception as e:
                print(f"Failed to load audio file: {e}")
                print("Generating synthetic signal instead...")
                self._generate_synthetic_signal(duration, sr)
        else:
            self._generate_synthetic_signal(duration, sr)
        
        print(f"Final signal shape: {self.original_signal.shape}")
        print(f"Sample rate: {self.sample_rate} Hz")
        return self.original_signal
    
    def _generate_synthetic_signal(self, duration, sr):
        """Generate synthetic noisy signal"""
        self.sample_rate = sr
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create clean signal (combination of sine waves)
        clean_signal = (np.sin(2 * np.pi * 440 * t) +  # A note
                       0.5 * np.sin(2 * np.pi * 880 * t) +  # A octave
                       0.3 * np.sin(2 * np.pi * 220 * t))   # A lower octave
        
        # Add various types of noise
        noise = (0.3 * np.random.normal(0, 1, len(t)) +  # White noise
                0.2 * np.sin(2 * np.pi * 60 * t) +       # 60Hz hum
                0.1 * np.sin(2 * np.pi * 120 * t))       # 120Hz harmonic
        
        self.original_signal = clean_signal + noise
        print(f"Generated synthetic noisy signal with duration {duration}s")
    
    def phase_02_preprocess_signal(self, frame_length=2048, hop_length=512):
        """
        Phase 02: Apply signal preprocessing including normalization and format conversion
        
        Args:
            frame_length (int): Length of each frame for STFT
            hop_length (int): Hop length for STFT
        """
        if LIBROSA_AVAILABLE:
            # Use librosa STFT
            stft_result = librosa.stft(self.original_signal, 
                                     n_fft=frame_length, 
                                     hop_length=hop_length)
            magnitude = np.abs(stft_result)
            # Convert to dB scale
            magnitude_db = librosa.amplitude_to_db(magnitude)
        else:
            # Use scipy STFT
            f, t, stft_result = signal.stft(self.original_signal,
                                          fs=self.sample_rate,
                                          nperseg=frame_length,
                                          noverlap=frame_length-hop_length)
            magnitude = np.abs(stft_result)
            # Convert to dB scale manually
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
        
        # Reshape for PCA (features x samples)
        self.feature_matrix = magnitude_db.T  # Transpose for sklearn format
        
        # Normalize the features
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        print(f"Preprocessed feature matrix shape: {self.feature_matrix_scaled.shape}")
        print(f"Features (frequency bins): {magnitude_db.shape[0]}")
        print(f"Time frames: {magnitude_db.shape[1]}")
        
        return self.feature_matrix_scaled
    
    def phase_03_apply_pca(self):
        """
        Phase 03: Implement PCA-based dimensionality reduction techniques
        """
        # Fit PCA to the preprocessed signal
        self.pca_features = self.pca.fit_transform(self.feature_matrix_scaled)
        
        # Get explained variance ratio
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"Reduced dimensions: {self.pca_features.shape}")
        print(f"Explained variance by first {self.n_components} components: "
              f"{cumulative_variance[-1]:.3f}")
        
        # Reconstruct signal in original space
        self.reconstructed_features = self.pca.inverse_transform(self.pca_features)
        self.reconstructed_features = self.scaler.inverse_transform(self.reconstructed_features)
        
        return self.pca_features, explained_variance
    
    def phase_04_visualize_subspace(self, save_plots=False):
        """
        Phase 04: Generate visual representations of the subspace mappings
        
        Args:
            save_plots (bool): Whether to save plots to files
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Original signal waveform
        time_axis = np.linspace(0, len(self.original_signal)/self.sample_rate, 
                               len(self.original_signal))
        axes[0, 0].plot(time_axis, self.original_signal)
        axes[0, 0].set_title('Original Noisy Signal')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # Plot 2: Original spectrogram
        if LIBROSA_AVAILABLE and LIBROSA_DISPLAY_AVAILABLE:
            D = librosa.stft(self.original_signal)
            DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img1 = librosa.display.specshow(DB, sr=self.sample_rate, 
                                           x_axis='time', y_axis='hz', 
                                           ax=axes[0, 1])
            plt.colorbar(img1, ax=axes[0, 1], format='%+2.0f dB')
        else:
            # Use scipy for spectrogram
            f, t, Sxx = signal.spectrogram(self.original_signal, self.sample_rate)
            DB = 10 * np.log10(Sxx + 1e-10)
            img1 = axes[0, 1].pcolormesh(t, f, DB, shading='gouraud', cmap='viridis')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Frequency (Hz)')
            plt.colorbar(img1, ax=axes[0, 1], format='%+2.0f dB')
        
        axes[0, 1].set_title('Original Spectrogram')
        
        # Plot 3: PCA explained variance
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        axes[0, 2].bar(range(1, len(explained_variance)+1), explained_variance)
        axes[0, 2].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'r-o')
        axes[0, 2].set_title('PCA Explained Variance')
        axes[0, 2].set_xlabel('Principal Component')
        axes[0, 2].set_ylabel('Explained Variance Ratio')
        axes[0, 2].legend(['Cumulative', 'Individual'])
        axes[0, 2].grid(True)
        
        # Plot 4: First 3 Principal Components
        axes[1, 0].plot(self.pca_features[:, :3])
        axes[1, 0].set_title('First 3 Principal Components')
        axes[1, 0].set_xlabel('Time Frame')
        axes[1, 0].set_ylabel('Component Value')
        axes[1, 0].legend(['PC1', 'PC2', 'PC3'])
        axes[1, 0].grid(True)
        
        # Plot 5: Reconstructed spectrogram
        reconstructed_spec = self.reconstructed_features.T
        img2 = axes[1, 1].imshow(reconstructed_spec, aspect='auto', 
                                origin='lower', cmap='viridis')
        axes[1, 1].set_title('PCA Reconstructed Spectrogram')
        axes[1, 1].set_xlabel('Time Frame')
        axes[1, 1].set_ylabel('Frequency Bin')
        plt.colorbar(img2, ax=axes[1, 1])
        
        # Plot 6: 2D PCA projection
        scatter = axes[1, 2].scatter(self.pca_features[:, 0], 
                                    self.pca_features[:, 1],
                                    c=range(len(self.pca_features)), 
                                    cmap='viridis', alpha=0.6)
        axes[1, 2].set_title('2D PCA Subspace Projection')
        axes[1, 2].set_xlabel('First Principal Component')
        axes[1, 2].set_ylabel('Second Principal Component')
        axes[1, 2].grid(True)
        plt.colorbar(scatter, ax=axes[1, 2], label='Time Frame')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('pca_audio_subspace_analysis.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'pca_audio_subspace_analysis.png'")
        
        plt.show()
        
        # Additional 3D visualization
        self._plot_3d_subspace()
    
    def _plot_3d_subspace(self):
        """Create 3D visualization of the subspace"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D scatter plot of first 3 principal components
        time_colors = np.arange(len(self.pca_features))
        scatter = ax.scatter(self.pca_features[:, 0], 
                           self.pca_features[:, 1], 
                           self.pca_features[:, 2],
                           c=time_colors, cmap='viridis', alpha=0.6)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D PCA Subspace Projection')
        
        # Add colorbar
        plt.colorbar(scatter, label='Time Frame')
        plt.show()
    
    def get_filtered_signal(self):
        """
        Reconstruct the filtered audio signal from PCA components
        
        Returns:
            numpy.ndarray: Filtered audio signal
        """
        # Convert back to magnitude spectrogram
        magnitude_filtered = self.reconstructed_features.T
        
        if LIBROSA_AVAILABLE:
            # Convert from dB back to linear scale
            magnitude_linear = librosa.db_to_amplitude(magnitude_filtered)
            
            # Use original phase information for reconstruction
            original_stft = librosa.stft(self.original_signal)
            original_phase = np.angle(original_stft)
            
            # Reconstruct complex spectrogram
            reconstructed_stft = magnitude_linear * np.exp(1j * original_phase)
            
            # Convert back to time domain
            self.processed_signal = librosa.istft(reconstructed_stft)
        else:
            # Use scipy for reconstruction
            magnitude_linear = 10**(magnitude_filtered / 20)
            
            # Get original STFT for phase
            f, t, original_stft = signal.stft(self.original_signal, 
                                            fs=self.sample_rate)
            original_phase = np.angle(original_stft)
            
            # Reconstruct complex spectrogram
            reconstructed_stft = magnitude_linear * np.exp(1j * original_phase)
            
            # Convert back to time domain
            _, self.processed_signal = signal.istft(reconstructed_stft, 
                                                   fs=self.sample_rate)
        
        # Ensure both signals have the same length
        min_length = min(len(self.original_signal), len(self.processed_signal))
        self.original_signal = self.original_signal[:min_length]
        self.processed_signal = self.processed_signal[:min_length]
        
        return self.processed_signal
    
    def analyze_noise_reduction(self):
        """
        Analyze the effectiveness of noise reduction
        """
        if self.processed_signal is None:
            self.get_filtered_signal()
        
        # Ensure signals have the same length (should already be handled in get_filtered_signal)
        min_length = min(len(self.original_signal), len(self.processed_signal))
        original_truncated = self.original_signal[:min_length]
        processed_truncated = self.processed_signal[:min_length]
        
        # Calculate power metrics
        original_power = np.mean(original_truncated**2)
        filtered_power = np.mean(processed_truncated**2)
        noise_power = np.mean((original_truncated - processed_truncated)**2)
        
        # Calculate SNR values (improved estimation)
        # Assume original signal has some noise, estimate clean signal power
        estimated_noise_level = 0.2  # Estimate based on our synthetic signal generation
        original_noise_power = original_power * estimated_noise_level
        original_clean_power = original_power * (1 - estimated_noise_level)
        
        original_snr = 10 * np.log10(original_clean_power / original_noise_power)
        
        # For filtered signal, noise is the reconstruction error
        if noise_power > 1e-10:  # Avoid division by zero
            filtered_snr = 10 * np.log10(filtered_power / noise_power)
            snr_improvement = filtered_snr - original_snr
        else:
            snr_improvement = float('inf')  # Perfect reconstruction
        
        # Calculate additional metrics
        mse = np.mean((original_truncated - processed_truncated)**2)
        correlation = np.corrcoef(original_truncated, processed_truncated)[0, 1]
        
        print(f"\nNoise Reduction Analysis:")
        print(f"Signal lengths - Original: {len(self.original_signal)}, Processed: {len(self.processed_signal)}")
        print(f"Original signal power: {original_power:.6f}")
        print(f"Filtered signal power: {filtered_power:.6f}")
        print(f"Reconstruction MSE: {mse:.6f}")
        print(f"Signal correlation: {correlation:.4f}")
        print(f"Estimated original SNR: {original_snr:.2f} dB")
        
        if noise_power > 1e-10:
            print(f"Filtered SNR: {filtered_snr:.2f} dB")
            print(f"SNR improvement: {snr_improvement:.2f} dB")
        else:
            print(f"SNR improvement: Perfect reconstruction (infinite)")
            snr_improvement = 50  # Cap for display purposes
        
        return snr_improvement

# Example usage and demonstration
def main():
    """
    Main function demonstrating the complete PCA audio subspace filtering workflow
    """
    print("=== PCA Audio Subspace Filtering Demo ===\n")
    
    # Get audio file path from user
    audio_file = input("Enter the path to your MP3 file (or press Enter for synthetic signal): ").strip()
    
    if audio_file and audio_file != "":
        # Remove quotes if user added them
        audio_file = audio_file.strip('"').strip("'")
        
        # Get duration preference
        try:
            duration_input = input("Enter duration to analyze in seconds (or press Enter for 10s): ").strip()
            duration = float(duration_input) if duration_input else 10.0
        except ValueError:
            duration = 10.0
            print("Invalid duration input, using 10 seconds")
        
        print(f"\nProcessing: {audio_file}")
        print(f"Duration: {duration} seconds")
    else:
        audio_file = None
        duration = 10.0
        print("Using synthetic signal for demonstration")
    
    # Initialize the filter
    audio_filter = AudioSubspaceFilter(n_components=15)
    
    # Execute the four-phase workflow
    print("\nPhase 01: Importing audio signal...")
    audio_filter.phase_01_import_signal(audio_file=audio_file, duration=duration)
    
    print("\nPhase 02: Preprocessing signal...")
    audio_filter.phase_02_preprocess_signal()
    
    print("\nPhase 03: Applying PCA dimensionality reduction...")
    pca_features, explained_variance = audio_filter.phase_03_apply_pca()
    
    print("\nPhase 04: Visualizing subspace mappings...")
    audio_filter.phase_04_visualize_subspace()
    
    # Get filtered signal and analyze results
    print("\nReconstructing filtered signal...")
    filtered_signal = audio_filter.get_filtered_signal()
    
    print("\nAnalyzing noise reduction effectiveness...")
    snr_improvement = audio_filter.analyze_noise_reduction()
    
    # Save filtered audio if original was loaded from file
    if audio_file:
        try:
            output_file = audio_file.replace('.mp3', '_filtered.wav').replace('.wav', '_filtered.wav')
            
            if LIBROSA_AVAILABLE:
                librosa.output.write_wav(output_file, filtered_signal, audio_filter.sample_rate)
                print(f"\nFiltered audio saved as: {output_file}")
            elif SOUNDFILE_AVAILABLE:
                sf.write(output_file, filtered_signal, audio_filter.sample_rate)
                print(f"\nFiltered audio saved as: {output_file}")
            else:
                print("\nCannot save filtered audio - no audio output library available")
        except Exception as e:
            print(f"\nCould not save filtered audio: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Original signal length: {len(audio_filter.original_signal)} samples")
    print(f"Duration processed: {len(audio_filter.original_signal)/audio_filter.sample_rate:.2f} seconds")
    print(f"PCA components used: {audio_filter.n_components}")
    print(f"Dimensionality reduction: {audio_filter.feature_matrix_scaled.shape[1]} → {audio_filter.n_components}")
    print(f"Variance explained: {np.sum(explained_variance):.3f}")
    print(f"SNR improvement: {snr_improvement:.2f} dB")
    
    if audio_file:
        print(f"Input file: {audio_file}")
        if 'output_file' in locals():
            print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()