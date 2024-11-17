import numpy as np
import matplotlib.pyplot as plt

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

# Load ECG signal data from file
path = 'ecg_standing.dat'
sampling_rate = 1000

def load_ecg_signal(filename, sampling_rate):
    try:
        ecg_signal = np.loadtxt(filename)
        return ecg_signal
    except ValueError:
        print('Error: Could not load file as text')
        return None
    
ecg_signal = load_ecg_signal(path, sampling_rate)

# Define a class for a real-time FIR filter
class FIRfilter:
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.buffer = np.zeros(len(coefficients))

    def dofilter(self, value):
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = value
        result = np.dot(self.coefficients, self.buffer)
        return result


# Create FIR filter coefficients
fir_coeffs = combined_highpass_bandstop_fir(sampling_rate)
FIR_filter = FIRfilter(fir_coeffs)
# filtering the ECG signal
filtered_ecg_signal = np.array([FIR_filter.dofilter(n) for n in ecg_signal])
# calculate the time axis
duration = len(ecg_signal) / sampling_rate
time = np.linspace(0, duration, len(ecg_signal))

# Plot the original and filtered signals
fig = plt.figure(figsize=(10, 5))

# Plot the original and filtered signals in the time domain
fig1 = fig.add_subplot(1, 2, 1)
fig1.plot(time, ecg_signal, label='Original ECG Signal', alpha=0.6)
fig1.plot(time, filtered_ecg_signal, label='Filtered ECG Signal (Real-time)', color='red', linewidth=1)
fig1.legend()
fig1.set_title('ECG Signal Before and After Real-time Filtering')
fig1.set_xlabel('Time [s]')
fig1.set_ylabel('Amplitude')

# Calculate and plot the frequency spectrum of the original and filtered signals
def do_fft(signal, sampling_rate):
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
    positive_freqs = freqs[:len(signal) // 2]
    positive_fft = fft_result[:len(signal) // 2]
    return positive_freqs, positive_fft

noise_freqs, noise_fft = do_fft(ecg_signal, sampling_rate)
filtered_freqs, filtered_fft = do_fft(filtered_ecg_signal, sampling_rate)

normalized_noise_fft = np.abs(noise_fft) / np.max(np.abs(noise_fft))
normalized_filtered_fft = np.abs(filtered_fft) / np.max(np.abs(filtered_fft))

# Plot the frequency spectrum of the original and filtered signals
fig2 = fig.add_subplot(1, 2, 2)
fig2.plot(noise_freqs, normalized_noise_fft, color = 'b',  label='Original Signal')
fig2.plot(filtered_freqs, normalized_filtered_fft, color = 'r',alpha=0.7, label='Filtered Signal')
fig2.set_title('Frequency Spectrum Before and After Filtering')
fig2.set_xlabel('Frequency [Hz]')
fig2.set_ylabel('Magnitude')
fig2.legend()
fig2.grid()

plt.tight_layout()
plt.savefig('./images/Task2.svg', format='svg')
plt.show()