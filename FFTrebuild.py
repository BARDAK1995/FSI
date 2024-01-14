import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


def rebuild_signal(data_series, target_freq_hz, timestep_duration, truncate_start=10000, truncate_end=30000, plot=False):
    """
    Rebuilds a signal from a pandas series using a target frequency.
    
    Parameters:
    - data_series: Pandas series containing the original data.
    - target_freq_hz: The frequency in Hz to use for rebuilding the signal.
    - timestep_duration: The duration of each timestep in seconds.
    - truncate_start: The starting index for truncating the data.
    - truncate_end: The ending index for truncating the data.
    - plot: Boolean flag to enable or disable plotting functionality.
    
    Returns:
    - amplitude: The amplitude of the rebuilt signal.
    - phase: The phase of the rebuilt signal in radians.
    """
    # Truncate the data
    truncated_data = data_series[truncate_start:truncate_end].reset_index(drop=True)
    
    # Convert truncated data to numpy array for FFT analysis
    truncated_data_array = truncated_data.to_numpy()
    
    # Perform FFT on the truncated data to obtain the frequency spectrum
    fft_result = fft(truncated_data_array)
    fft_frequencies = np.fft.fftfreq(len(truncated_data_array), d=timestep_duration)
    
    # Locate the index of the target frequency in the FFT result
    index_target = np.argmin(np.abs(fft_frequencies - target_freq_hz))
    
    # Extract the complex amplitude of the target frequency
    complex_amplitude = fft_result[index_target]
    
    # Calculate the peak amplitude and phase of the sine wave at the target frequency
    amplitude = np.abs(complex_amplitude) * 2 / len(truncated_data_array)
    phase = np.angle(complex_amplitude)
    
    # Time vector for the truncated data
    time_vector = np.arange(truncate_start, truncate_end) * timestep_duration
    
    # Rebuild the signal with the target frequency component including the phase
    rebuilt_signal_with_phase = amplitude * np.cos(2 * np.pi * target_freq_hz * time_vector + phase)
    std_dev = np.std(truncated_data_array)  # Calculate the standard deviation

    if plot:
        # Plot the original data and the rebuilt signal with phase correction for comparison
        plt.figure(figsize=(12, 6))
        plt.plot(time_vector / timestep_duration, truncated_data_array, label='Original Data', linewidth=0.5)
        plt.plot(time_vector / timestep_duration, rebuilt_signal_with_phase + truncated_data.mean(), label=f'Rebuilt Signal ({target_freq_hz} Hz)', alpha=0.7,linewidth=4)
        plt.title(f'Original Data and Phase-Corrected Rebuilt Signal ({target_freq_hz} Hz) Overlay')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return amplitude, phase, std_dev


caseName = "100at5"
# point = 7
# file_path = f"./PM_BC_december/{caseName}/Point{str(point)}.dat"
# data = pd.read_csv(file_path, delim_whitespace=True, header=None)
# amplitude_t, phase_t, stdDev_t = rebuild_signal(data.iloc[:, 13], 100e3, 5e-9, plot=True)


strlist_p = []
phaselist_p = []
stdDevList_p= []

strlist_t = []
phaselist_t = []
stdDevList_t= []

strlist_r = []
phaselist_r = []
stdDevList_r= []
for point in range(1, 11):
    file_path = f"./PM_BC_december/{caseName}/Point{str(point)}.dat"
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    amplitude_r, phase_r, stdDev_r = rebuild_signal(data.iloc[:, 9], 100e3, 5e-9, plot=False)
    amplitude_t, phase_t, stdDev_t = rebuild_signal(data.iloc[:, 13], 100e3, 5e-9, plot=True)
    amplitude_p, phase_p, stdDev_p = rebuild_signal(data.iloc[:, 16], 100e3, 5e-9, plot=False)
    strlist_r.append(amplitude_r)
    phaselist_r.append(180 - phase_r*180/3.14)
    stdDevList_r.append(stdDev_r)

    strlist_p.append(amplitude_p)
    phaselist_p.append(180 - phase_p*180/3.14)
    stdDevList_p.append(stdDev_p)

    strlist_t.append(amplitude_t)
    phaselist_t.append(180 - phase_t*180/3.14)
    stdDevList_t.append(stdDev_t)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

# Plot strlist in the top subplot
ax1_r = ax1.twinx()
ax1_r.plot([i+1 for i in range(len(strlist_r))], strlist_r, '-', color='gray', linewidth=1, label='Density')
ax1.plot([i+1 for i in range(len(strlist_t))], strlist_t, '-', color='red', linewidth=1, label='Temperature')
ax1.plot([i+1 for i in range(len(strlist_p))], strlist_p, '-', color='blue', linewidth=1, label='Pressure')
ax1.set_ylabel('Perturb AMPLITUDE')
ax1_r.set_ylabel('Density', color='gray')
ax1.legend(loc='upper left')
ax1_r.legend(loc='upper right')
ax1.grid(axis='x')

# Plot phaselist in the middle subplot
ax2.plot([i+1 for i in range(len(phaselist_r))], phaselist_r, '-', color='gray', linewidth=1, label='Density')
ax2.plot([i+1 for i in range(len(phaselist_t))], phaselist_t, '-', color='red', linewidth=1, label='Temperature')
ax2.plot([i+1 for i in range(len(phaselist_p))], phaselist_p, '-', color='blue', linewidth=1, label='Pressure')
ax2.set_ylabel('Phase (degrees)')
ax2.legend(loc='upper left')
ax2.grid(axis='x')

# Plot stdDevList in the bottom subplot
ax3_r = ax3.twinx()
ax3_r.plot([i+1 for i in range(len(stdDevList_r))], stdDevList_r, '-', color='gray', linewidth=1, label='Density')
ax3.plot([i+1 for i in range(len(stdDevList_t))], stdDevList_t, '-', color='red', linewidth=1, label='Temperature')
ax3.plot([i+1 for i in range(len(stdDevList_p))], stdDevList_p, '-', color='blue', linewidth=1, label='Pressure')
ax3.set_xlabel('Probe Index')
ax3.set_ylabel('STD dev')
ax3_r.set_ylabel('Density', color='gray')
ax3.legend(loc='upper left')
ax3_r.legend(loc='upper right')
ax3.grid(axis='x')

# Set xticks
ax3.set_xticks([i+1 for i in range(len(stdDevList_r))])

plt.tight_layout()

file_pathplot = f"./PM_BC_december/{caseName}/plot.png"
plt.savefig(file_pathplot)
plt.show()