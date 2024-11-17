import numpy as np
import matplotlib.pyplot as plt

# Function to generate 50Hz reference signal with DC
def generate_reference_signals(sample_index, sampling_rate):
    dc_component = 1.0  # Constant for DC reference
    sin_50hz = np.sin(2 * np.pi * 50 * sample_index / sampling_rate)
    return dc_component, sin_50hz

def combined_highpass_bandstop_fir(sampling_rate, hp_cutoff=0.5, bs_center=50, bs_bandwidth=2):
    
    # Define Nyquist frequency and normalized cutoff
    nyquist = sampling_rate / 2
    normalized_hp_cutoff = hp_cutoff / nyquist
    
    # Calculate the high-pass filter order
    hp_transition_band = 0.5  # Example transition bandwidth
    hp_order = int(hp_transition_band / (normalized_hp_cutoff))  # Using Bt/N = (Ws - Wp) formula
    if hp_order % 2 == 0:  # Ensure order is odd for symmetry
        hp_order += 1
    highpass_coeffs = np.sinc(2 * normalized_hp_cutoff * (np.arange(hp_order) - (hp_order - 1) / 2))
    highpass_coeffs *= np.hamming(hp_order)
    highpass_coeffs = -highpass_coeffs
    highpass_coeffs[(hp_order - 1) // 2] += 1  # Make it a high-pass filter

    # Calculate the band-stop filter order
    bs_transition_band = 0.5  # Example transition bandwidth for the band-stop filter
    normalized_bs_low_cutoff = (bs_center - bs_bandwidth / 2) / nyquist
    normalized_bs_high_cutoff = (bs_center + bs_bandwidth / 2) / nyquist
    bs_order = int(bs_transition_band / (normalized_bs_high_cutoff - normalized_bs_low_cutoff))
    if bs_order % 2 == 0:
        bs_order += 1
    bandstop_coeffs = np.sinc(2 * normalized_bs_high_cutoff * (np.arange(bs_order) - (bs_order - 1) / 2)) \
                      - np.sinc(2 * normalized_bs_low_cutoff * (np.arange(bs_order) - (bs_order - 1) / 2))
    bandstop_coeffs *= np.hamming(bs_order)
    bandstop_coeffs[(bs_order - 1) // 2] += 1

    # Combine high-pass and band-stop filter coefficients by convolution
    combined_coeffs = np.convolve(highpass_coeffs, bandstop_coeffs)
    combined_coeffs /= np.sum(combined_coeffs)  # Normalize filter coefficients
    
    return combined_coeffs

# Define a class for adaptive FIR filter
class FIRFilter:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.weights = np.zeros(2)
        self.buffer = np.zeros(len(coefficients))

    def dofilter(self, value):
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = value
        result = np.dot(self.coefficients, self.buffer)
        return result


    def doFilterAdaptive(self, signal, referance_signal, learningRate):
              
        # set the reference signal
        dc_reference, sine_50hz = referance_signal 
        reference = np.array([dc_reference, sine_50hz])
        
        # Update the adaptive filter weights
        adaptive = np.dot(self.weights, reference)
        error = signal - adaptive
        
        # Update weights using LMS rule: w = w + 2 * learningRate * error * reference
        self.weights += 2 * learningRate * error * reference
        
        return error

# Load ECG signal data from file
def load_ecg_signal(filename, sampling_rate):
    try:
        ecg_signal = np.loadtxt(filename)
        return ecg_signal
    except ValueError:
        print('Error: Could not load file as text')
        return None

# set parameters
path = 'ecg_standing.dat'
sampling_rate = 1000
learning_rate = 0.05
n_taps = 20

# Load the ECG signal
ecg_signal = load_ecg_signal(path, sampling_rate)
duration = len(ecg_signal) / sampling_rate
time = np.linspace(0, duration, len(ecg_signal))
fir_filter = FIRFilter(coefficients=np.ones(n_taps) / n_taps)   # Initialize FIR filter
adaptive = [] # initialize the adaptive output

# Iterate over each sample in the ECG signal
for n, sample in enumerate(ecg_signal):
    dc_ref, sine_50hz_ref = generate_reference_signals(n, sampling_rate)
    filtered_adaptive = fir_filter.doFilterAdaptive(sample, (dc_ref, sine_50hz_ref), learning_rate)
    adaptive.append(filtered_adaptive)

# Create FIR filter coefficients
fir_coeffs = combined_highpass_bandstop_fir(sampling_rate)
FIR_filter = FIRFilter(fir_coeffs)
# filtering the ECG signal
filtered_ecg_signal = np.array([FIR_filter.dofilter(n) for n in ecg_signal])

# Convert to numpy arrays for plotting
adaptive_output = np.array(adaptive)

# Plot the original noisy ECG, adaptive filter output, and FIR filter output
plt.figure(figsize=(12, 10))

# Plot in the time domain
plt.subplot(2, 1, 1)
plt.plot(time, ecg_signal, label='Noisy ECG', alpha=0.5)
plt.plot(time, adaptive_output, label='Adaptive LMS Filter Output', color='red', linewidth=1)
plt.plot(time, filtered_ecg_signal, label='Traditional FIR Filter Output', color='blue', linewidth=1)
plt.legend()
plt.grid()
plt.xlabel('time [s]')
plt.ylabel('Amplitude')
plt.title('ECG Signal Filtering Comparison')

# Calculate and plot the frequency spectrum
def do_fft(signal, sampling_rate):
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
    positive_freqs = freqs[:len(signal) // 2]
    positive_fft = fft_result[:len(signal) // 2]
    return positive_freqs, positive_fft

adaptive_freqs, adaptive_fft = do_fft(adaptive_output, sampling_rate)
filtered_freqs, filtered_fft = do_fft(filtered_ecg_signal, sampling_rate)
original_freqs, original_fft = do_fft(ecg_signal, sampling_rate)

normalized_adaptive_fft = np.abs(adaptive_fft) / np.max(np.abs(adaptive_fft))
normalized_filtered_fft = np.abs(filtered_fft) / np.max(np.abs(filtered_fft))
normalized_original_fft = np.abs(original_fft) / np.max(np.abs(original_fft))

# plot the frequency spectrum
plt.subplot(2, 1, 2)
plt.plot(original_freqs, normalized_original_fft, color = 'b',  label='Original Signal')
plt.plot(filtered_freqs, normalized_filtered_fft, color = 'r',alpha=0.7, label='FIR Filter Output')
plt.plot(adaptive_freqs, normalized_adaptive_fft, color = 'g',alpha=0.7, label='Adaptive Filter Output')
plt.title('Frequency Spectrum Before and After Filtering')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('./images/Task3.svg', format='svg')
plt.show()
