import numpy as np
import matplotlib.pyplot as plt
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

class AudioSVDFilter:
    def __init__(self, n_components=10):
        """
        Initialize the Audio SVD Filter
        
        Args:
            n_components (int): Number of singular values/vectors to retain
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.original_signal = None
        self.processed_signal = None
        self.sample_rate = None
        self.U = None  # Left singular vectors
        self.S = None  # Singular values
        self.Vt = None  # Right singular vectors (transposed)
        
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
    
    def phase_01_extract_features(self, audio_file=None, duration=10, sr=22050):
        """
        Phase 01: Extract features from the audio signal
        
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
    
    def phase_02_apply_svd(self, frame_length=2048, hop_length=512):
        """
        Phase 02: Apply SVD to the feature matrix
        
        Args:
            frame_length (int): Length of each frame for STFT
            hop_length (int): Hop length for STFT
        """
        # Convert to STFT representation (same as PCA preprocessing)
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
        
        # Prepare feature matrix (frequency bins x time frames)
        self.feature_matrix = magnitude_db
        
        # Normalize the features
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix.T).T
        
        # Apply SVD decomposition
        print("Applying SVD decomposition...")
        self.U, self.S, self.Vt = np.linalg.svd(self.feature_matrix_scaled, full_matrices=False)
        
        # Keep only the first n_components
        self.U_reduced = self.U[:, :self.n_components]
        self.S_reduced = self.S[:self.n_components]
        self.Vt_reduced = self.Vt[:self.n_components, :]
        
        print(f"Original feature matrix shape: {self.feature_matrix_scaled.shape}")
        print(f"SVD shapes - U: {self.U_reduced.shape}, S: {self.S_reduced.shape}, Vt: {self.Vt_reduced.shape}")
        print(f"Singular values (first 10): {self.S_reduced[:10]}")
        
        return self.U_reduced, self.S_reduced, self.Vt_reduced
    
    def phase_03_project_to_subspace(self):
        """
        Phase 03: Use the singular vectors to project the audio signal into a lower-dimensional subspace
        """
        # Reconstruct using reduced SVD components
        self.reconstructed_matrix = np.dot(self.U_reduced * self.S_reduced, self.Vt_reduced)
        
        # Transform back from scaled space
        self.reconstructed_features = self.scaler.inverse_transform(self.reconstructed_matrix.T).T
        
        # Calculate reconstruction error
        reconstruction_error = np.mean((self.feature_matrix - self.reconstructed_features)**2)
        
        # Calculate energy preservation
        original_energy = np.sum(self.S**2)
        preserved_energy = np.sum(self.S_reduced**2)
        energy_ratio = preserved_energy / original_energy
        
        print(f"Reconstruction completed")
        print(f"Reconstruction MSE: {reconstruction_error:.6f}")
        print(f"Energy preserved: {energy_ratio:.3f} ({energy_ratio*100:.1f}%)")
        print(f"Components used: {self.n_components} out of {len(self.S)}")
        
        return self.reconstructed_features, energy_ratio
    
    def phase_04_visualize_results(self, save_plots=False):
        """
        Phase 04: Visualize the results of the projection
        
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
        
        # Plot 3: Singular values
        axes[0, 2].semilogy(self.S, 'b-o', markersize=4)
        axes[0, 2].semilogy(self.S_reduced, 'r-s', markersize=6)
        axes[0, 2].axvline(x=self.n_components-0.5, color='r', linestyle='--', alpha=0.7)
        axes[0, 2].set_title('Singular Values (Log Scale)')
        axes[0, 2].set_xlabel('Component Index')
        axes[0, 2].set_ylabel('Singular Value')
        axes[0, 2].legend(['All Singular Values', f'First {self.n_components} Components', 'Cutoff'])
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Left singular vectors (U matrix - first few components)
        for i in range(min(5, self.n_components)):
            axes[1, 0].plot(self.U_reduced[:, i], label=f'U_{i+1}', alpha=0.8)
        axes[1, 0].set_title('Left Singular Vectors (Frequency Domain)')
        axes[1, 0].set_xlabel('Frequency Bin')
        axes[1, 0].set_ylabel('Component Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Right singular vectors (V matrix - first few components)
        for i in range(min(5, self.n_components)):
            axes[1, 1].plot(self.Vt_reduced[i, :], label=f'V_{i+1}', alpha=0.8)
        axes[1, 1].set_title('Right Singular Vectors (Time Domain)')
        axes[1, 1].set_xlabel('Time Frame')
        axes[1, 1].set_ylabel('Component Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Reconstructed spectrogram
        img2 = axes[1, 2].imshow(self.reconstructed_features, aspect='auto', 
                                origin='lower', cmap='viridis')
        axes[1, 2].set_title(f'SVD Reconstructed Spectrogram ({self.n_components} components)')
        axes[1, 2].set_xlabel('Time Frame')
        axes[1, 2].set_ylabel('Frequency Bin')
        plt.colorbar(img2, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('svd_audio_subspace_analysis.png', dpi=300, bbox_inches='tight')
            print("Plots saved as 'svd_audio_subspace_analysis.png'")
        
        plt.show()
        
        # Additional visualization: Energy distribution
        self._plot_energy_analysis()
    
    def _plot_energy_analysis(self):
        """Create energy analysis visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Cumulative energy
        cumulative_energy = np.cumsum(self.S**2) / np.sum(self.S**2)
        ax1.plot(cumulative_energy, 'b-', linewidth=2)
        ax1.axvline(x=self.n_components-1, color='r', linestyle='--', 
                   label=f'{self.n_components} components')
        ax1.axhline(y=cumulative_energy[self.n_components-1], color='r', 
                   linestyle=':', alpha=0.7)
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cumulative Energy Ratio')
        ax1.set_title('Cumulative Energy Preservation')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Energy contribution per component
        energy_contribution = (self.S**2) / np.sum(self.S**2)
        bars = ax2.bar(range(min(20, len(energy_contribution))), 
                      energy_contribution[:min(20, len(energy_contribution))])
        # Highlight selected components
        for i in range(min(self.n_components, 20)):
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
        ax2.set_xlabel('Component Index')
        ax2.set_ylabel('Energy Contribution')
        ax2.set_title('Energy Contribution per Component')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_filtered_signal(self):
        """
        Reconstruct the filtered audio signal from SVD components
        
        Returns:
            numpy.ndarray: Filtered audio signal
        """
        # Convert back to magnitude spectrogram
        magnitude_filtered = self.reconstructed_features
        
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
    
    def analyze_filtering_performance(self):
        """
        Analyze the effectiveness of SVD filtering
        """
        if self.processed_signal is None:
            self.get_filtered_signal()
        
        # Ensure signals have the same length
        min_length = min(len(self.original_signal), len(self.processed_signal))
        original_truncated = self.original_signal[:min_length]
        processed_truncated = self.processed_signal[:min_length]
        
        # Calculate performance metrics
        original_power = np.mean(original_truncated**2)
        filtered_power = np.mean(processed_truncated**2)
        noise_power = np.mean((original_truncated - processed_truncated)**2)
        
        # Calculate SNR values
        estimated_noise_level = 0.2
        original_noise_power = original_power * estimated_noise_level
        original_clean_power = original_power * (1 - estimated_noise_level)
        
        original_snr = 10 * np.log10(original_clean_power / original_noise_power)
        
        if noise_power > 1e-10:
            filtered_snr = 10 * np.log10(filtered_power / noise_power)
            snr_improvement = filtered_snr - original_snr
        else:
            snr_improvement = float('inf')
        
        # Additional SVD-specific metrics
        mse = np.mean((original_truncated - processed_truncated)**2)
        correlation = np.corrcoef(original_truncated, processed_truncated)[0, 1]
        energy_preserved = np.sum(self.S_reduced**2) / np.sum(self.S**2)
        
        print(f"\nSVD Filtering Performance Analysis:")
        print(f"Signal lengths - Original: {len(self.original_signal)}, Processed: {len(self.processed_signal)}")
        print(f"Components used: {self.n_components} out of {len(self.S)} ({self.n_components/len(self.S)*100:.1f}%)")
        print(f"Energy preserved: {energy_preserved:.3f} ({energy_preserved*100:.1f}%)")
        print(f"Original signal power: {original_power:.6f}")
        print(f"Filtered signal power: {filtered_power:.6f}")
        print(f"Reconstruction MSE: {mse:.6f}")
        print(f"Signal correlation: {correlation:.4f}")
        print(f"Estimated original SNR: {original_snr:.2f} dB")
        
        if noise_power > 1e-10:
            print(f"Filtered SNR: {filtered_snr:.2f} dB")
            print(f"SNR improvement: {snr_improvement:.2f} dB")
        else:
            print(f"SNR improvement: Perfect reconstruction")
            snr_improvement = 50
        
        return snr_improvement, energy_preserved

def main():
    """
    Main function demonstrating the complete SVD audio subspace filtering workflow
    """
    print("=== SVD Audio Subspace Filtering Demo ===\n")
    
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
        
        # Get number of components
        try:
            components_input = input("Enter number of SVD components to keep (or press Enter for 15): ").strip()
            n_components = int(components_input) if components_input else 15
        except ValueError:
            n_components = 15
            print("Invalid components input, using 15 components")
        
        print(f"\nProcessing: {audio_file}")
        print(f"Duration: {duration} seconds")
        print(f"SVD components: {n_components}")
    else:
        audio_file = None
        duration = 10.0
        n_components = 15
        print("Using synthetic signal for demonstration")
    
    # Initialize the SVD filter
    svd_filter = AudioSVDFilter(n_components=n_components)
    
    # Execute the four-phase workflow
    print("\n" + "="*50)
    print("Phase 01: Extracting features from audio signal...")
    svd_filter.phase_01_extract_features(audio_file=audio_file, duration=duration)
    
    print("\n" + "="*50)
    print("Phase 02: Applying SVD to feature matrix...")
    U, S, Vt = svd_filter.phase_02_apply_svd()
    
    print("\n" + "="*50)
    print("Phase 03: Projecting to lower-dimensional subspace...")
    reconstructed_features, energy_ratio = svd_filter.phase_03_project_to_subspace()
    
    print("\n" + "="*50)
    print("Phase 04: Visualizing projection results...")
    svd_filter.phase_04_visualize_results()
    
    # Reconstruct and analyze filtered signal
    print("\n" + "="*50)
    print("Reconstructing filtered audio signal...")
    filtered_signal = svd_filter.get_filtered_signal()
    
    print("Analyzing filtering performance...")
    snr_improvement, energy_preserved = svd_filter.analyze_filtering_performance()
    
    # Save filtered audio if original was loaded from file
    if audio_file:
        try:
            output_file = audio_file.replace('.mp3', '_svd_filtered.wav').replace('.wav', '_svd_filtered.wav')
            
            if LIBROSA_AVAILABLE:
                librosa.output.write_wav(output_file, filtered_signal, svd_filter.sample_rate)
                print(f"\nSVD filtered audio saved as: {output_file}")
            elif SOUNDFILE_AVAILABLE:
                sf.write(output_file, filtered_signal, svd_filter.sample_rate)
                print(f"\nSVD filtered audio saved as: {output_file}")
            else:
                print("\nCannot save filtered audio - no audio output library available")
        except Exception as e:
            print(f"\nCould not save filtered audio: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("=== SVD FILTERING SUMMARY ===")
    print("="*60)
    print(f"Original signal length: {len(svd_filter.original_signal)} samples")
    print(f"Duration processed: {len(svd_filter.original_signal)/svd_filter.sample_rate:.2f} seconds")
    print(f"SVD components used: {svd_filter.n_components} out of {len(svd_filter.S)}")
    print(f"Dimensionality reduction: {len(svd_filter.S)} → {svd_filter.n_components}")
    print(f"Energy preserved: {energy_preserved:.3f} ({energy_preserved*100:.1f}%)")
    print(f"SNR improvement: {snr_improvement:.2f} dB")
    print(f"Compression ratio: {len(svd_filter.S)/svd_filter.n_components:.1f}:1")
    
    if audio_file:
        print(f"Input file: {audio_file}")
        if 'output_file' in locals():
            print(f"Output file: {output_file}")
    
    print("="*60)

if __name__ == "__main__":
    main()