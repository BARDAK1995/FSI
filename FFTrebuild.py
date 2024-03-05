import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


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


caseName = "500hzat5"
# caseName = "ref_edge"

caseNameREF = "ref"
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

phaselist_pRef = []
phaselist_tRef = []
strlist_pRef = []
strlist_tRef = []

strlist_r = []
phaselist_r = []
stdDevList_r= []
Targetfreq=500e3

for point in range(1, 90):
    file_path = f"./PM_feb/{caseName}/PROBE_{str(point)}"
    file_path_ref = f"./PM_feb/{caseNameREF}/PROBE_{str(point)}"
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    dataref = pd.read_csv(file_path_ref, delim_whitespace=True, header=None)
    if(point==40):
        plotflag=True
    else:
        plotflag=False
    densityDATA= data.iloc[:, 9]
    temperatureDATA= data.iloc[:, 13]
    pressureDATA= data.iloc[:, 16]
    amplitude_r, phase_r, stdDev_r = rebuild_signal(densityDATA, Targetfreq, 5e-9, plot=plotflag)
    amplitude_t, phase_t, stdDev_t = rebuild_signal(temperatureDATA, Targetfreq, 5e-9, plot=plotflag)
    amplitude_p, phase_p, stdDev_p = rebuild_signal(pressureDATA, Targetfreq, 5e-9, plot=plotflag)
    
    amplitude_tRef, phase_tRef, stdDev_tRef = rebuild_signal(dataref.iloc[:, 13], Targetfreq, 5e-9, plot=False)
    amplitude_pRef, phase_pRef, stdDev_pRef = rebuild_signal(dataref.iloc[:, 16], Targetfreq, 5e-9, plot=False)/np.mean(dataref.iloc[:, 16])*100

    
    strlist_r.append(amplitude_r/np.mean(densityDATA)*100)
    phaselist_r.append(180 - phase_r*180/3.14)
    stdDevList_r.append(stdDev_r)

    strlist_p.append(amplitude_p/np.mean(pressureDATA)*100)
    phaselist_p.append(180 - phase_p*180/3.14)
    stdDevList_p.append(stdDev_p)

    strlist_t.append(amplitude_t/np.mean(temperatureDATA)*100)
    phaselist_t.append(180 - phase_t*180/3.14)
    stdDevList_t.append(stdDev_t)

    strlist_tRef.append(amplitude_tRef/np.mean(dataref.iloc[:, 13])*100)
    phaselist_tRef.append(180 - phase_tRef*180/3.14)

    strlist_pRef.append(amplitude_pRef/np.mean(dataref.iloc[:, 16])*100)
    phaselist_pRef.append(180 - phase_pRef*180/3.14)

# fig, ax1 = plt.subplots(1, 1, figsize=(10, 3), sharex=True)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# window_size = 3  # You can adjust this value as needed
# kernel = np.ones(window_size) / window_size
# strlist_r = np.convolve(strlist_r, kernel, mode='valid')

sigma = 1  # Adjust this value to control the amount of smoothing
# Apply a Gaussian filter for a non-uniform weighted moving average
strlist_r = gaussian_filter1d(strlist_r, sigma=sigma)
strlist_t = gaussian_filter1d(strlist_t, sigma=sigma)
strlist_p = gaussian_filter1d(strlist_p, sigma=sigma)
strlist_tRef = gaussian_filter1d(strlist_tRef, sigma=sigma)

stdDevList_r = gaussian_filter1d(stdDevList_r, sigma=sigma*2)
stdDevList_t = gaussian_filter1d(stdDevList_t, sigma=sigma*2)
stdDevList_p = gaussian_filter1d(stdDevList_p, sigma=sigma*2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

# Plot strlist in the top subplot
ax1_r = ax1.twinx()
ax1_r.plot([(-0.0 + i*15/90) for i in range(len(strlist_r))], strlist_r, '-', color='black', linewidth=1, label='Density')
ax1.plot([(-0.0 + i*15/90)  for i in range(len(strlist_t))], strlist_t, '-', color='red', linewidth=1, label='Temperature')
ax1.plot([(-0.0 + i*15/90) for i in range(len(strlist_p))], strlist_p, '-', color='blue', linewidth=1, label='Pressure')

ax1.plot([(-0.0 + i*15/90)  for i in range(len(strlist_tRef))], strlist_tRef, '--', color='purple', linewidth=0.5, label='TemperatureREF')


ax1.set_ylabel('Perturb AMPLITUDE')
# ax1.set_xlabel('Location (mm)')

ax1_r.set_ylabel('Density', color='gray')
ax1.legend(loc='upper left')
ax1_r.legend(loc='upper right')
ax1.grid(axis='x')

# Plot phaselist in the middle subplot
# ax2.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_r))], phaselist_r, '-', color='gray', linewidth=1, label='Density')
# ax2.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_t))], phaselist_t, '-', color='red', linewidth=1, label='Temperature')
ax2.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_p))], phaselist_p, '-', color='blue', linewidth=1, label='Pressure')
# ax2.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_tRef))], phaselist_tRef, '--', color='black', linewidth=1, label='Temperature_REF')
# ax2.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_pRef))], phaselist_pRef, '--', color='blue', linewidth=1, label='Temperature_REF')


ax2.set_ylabel('Phase (degrees)')
ax2.legend(loc='upper left')
ax2.grid(axis='x')

# Plot stdDevList in the bottom subplot
ax3_r = ax3.twinx()
ax3_r.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_r))], stdDevList_r, '-', color='gray', linewidth=1, label='Density')
ax3.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_t))], stdDevList_t, '-', color='red', linewidth=1, label='Temperature')
ax3.plot([(-0.0 + i*15/90) for i in range(len(stdDevList_p))], stdDevList_p, '-', color='blue', linewidth=1, label='Pressure')
ax3.set_xlabel('Probe Location(mm)')
ax3.set_ylabel('STD dev')
ax3_r.set_ylabel('Density', color='gray')
ax3.legend(loc='upper left')
ax3_r.legend(loc='upper right')
ax3.grid(axis='x')

# Set xticks
# ax3.set_xticks([(-0.0 + i*15/90) for i in range(len(stdDevList_r))])
ax3.set_xticks(np.linspace(0, 14, 15))
# ax1.set_xticks([(-1.5 + i*1.5) for i in range(len(stdDevList_r))])

plt.tight_layout()

file_pathplot = f"./PM_feb/{caseName}/plot_{Targetfreq}.png"
plt.savefig(file_pathplot)
plt.show()