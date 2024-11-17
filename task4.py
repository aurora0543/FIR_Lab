import numpy as np
import matplotlib.pyplot as plt

# Load ECG signal data from file
def load_ecg_signal(filename, sampling_rate):
    try:
        ecg_signal = np.loadtxt(filename)
        return ecg_signal
    except ValueError:
        print('Error: Could not load file as text')
        return None

# Define a function to detect peaks
def detect_peaks(signal, min_distance, window_size):
    peaks = []
    for i in range(0, len(signal), window_size):
        local_signal = signal[i:i + window_size]
        local_threshold = np.mean(local_signal) + 2 * np.std(local_signal)
        for j in range(1, len(local_signal) - 1):
            if local_signal[j] > local_threshold and local_signal[j] > local_signal[j - 1] and local_signal[j] > local_signal[j + 1]:
                if len(peaks) == 0 or (i + j - peaks[-1]) > min_distance:
                    peaks.append(i + j)
    return np.array(peaks)

if __name__ == "__main__":
    # Load ECG signals
    fs = 1000
    clean_ecg = load_ecg_signal('ecg_lying.dat', fs)
    noisy_ecg = load_ecg_signal('ecg_standing.dat', fs)

    clean_threshold = np.mean(clean_ecg) + 2 * np.std(clean_ecg)
    clean_peaks = detect_peaks(clean_ecg, min_distance=int(fs * 0.6), window_size=fs // 2)
    start_idx = clean_peaks[0] - 25 # 25 samples before the first peak
    end_idx = clean_peaks[0] + 25
    r_peak_template = clean_ecg[start_idx:end_idx]

    # Perform manual convolution
    filtered_ecg = np.convolve(noisy_ecg, r_peak_template)

    # detect the R-pe
    threshold = np.mean(filtered_ecg) + 2 * np.std(filtered_ecg)
    r_peaks = detect_peaks(filtered_ecg, min_distance=int(fs * 0.6), window_size=fs // 2)

    # calculate the heart rate
    r_intervals = np.diff(r_peaks) / fs  
    heart_rate = 60 / r_intervals  
    time_points = r_peaks[:-1] / fs

    # soothe the heart rate
    heart_rate_smooth = np.convolve(heart_rate, np.ones(5) / 5, mode='valid')

    
    plt.figure(figsize=(12, 8))

    # Plot the filtered ECG signal with detected R-peaks
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(filtered_ecg)) / fs, filtered_ecg, label="Filtered ECG")
    plt.scatter(r_peaks / fs, filtered_ecg[r_peaks], color='red', label="Detected R-peaks", zorder=5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Filtered ECG Signal with Detected R-peaks")
    plt.legend()
    plt.grid(True)

    #  Plot the heart rate over time
    plt.subplot(2, 1, 2)
    plt.plot(time_points[:len(heart_rate_smooth)], heart_rate_smooth, label="Heart Rate (BPM)", marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title("Momentary Heart Rate Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./images/Task4.svg', format='svg')
    plt.show()