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


caseName = "100hz_at5_edge"
# caseName = "ref_edge"

caseNameREF = "ref_edge"
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
phaselist_rRef = []

strlist_pRef = []
strlist_tRef = []
strlist_rRef = []


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
    amplitude_pRef, phase_pRef, stdDev_pRef = rebuild_signal(dataref.iloc[:, 16], Targetfreq, 5e-9, plot=False)
    amplitude_rRef, phase_rRef, stdDev_rRef = rebuild_signal(dataref.iloc[:, 9], Targetfreq, 5e-9, plot=False)

    
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

    strlist_rRef.append(amplitude_rRef/np.mean(dataref.iloc[:, 9])*100)
    phaselist_rRef.append(180 - phase_rRef*180/3.14)

# fig, ax1 = plt.subplots(1, 1, figsize=(10, 3), sharex=True)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# window_size = 3  # You can adjust this value as needed
# kernel = np.ones(window_size) / window_size
# strlist_r = np.convolve(strlist_r, kernel, mode='valid')

sigma = 0.8  # Adjust this value to control the amount of smoothing
# Apply a Gaussian filter for a non-uniform weighted moving average
strlist_r = gaussian_filter1d(strlist_r, sigma=sigma)
strlist_t = gaussian_filter1d(strlist_t, sigma=sigma)
strlist_p = gaussian_filter1d(strlist_p, sigma=sigma)
strlist_tRef = gaussian_filter1d(strlist_tRef, sigma=sigma)
strlist_pRef = gaussian_filter1d(strlist_pRef, sigma=sigma)
strlist_rRef = gaussian_filter1d(strlist_tRef, sigma=sigma)


stdDevList_r = gaussian_filter1d(stdDevList_r, sigma=sigma*2)
stdDevList_t = gaussian_filter1d(stdDevList_t, sigma=sigma*2)
stdDevList_p = gaussian_filter1d(stdDevList_p, sigma=sigma*2)

phaselist_r = np.array(phaselist_r)
phaselist_rRef = np.array(phaselist_rRef)
phaselist_t = np.array(phaselist_t)
phaselist_tRef = np.array(phaselist_tRef)

phaselist_p = np.array(phaselist_p)
phaselist_pRef = np.array(phaselist_pRef)

# phaselist_r[phaselist_r > 350] = 0
# Define the thresholds
high_threshold = 320
neighbor_threshold = 50

# Iterate through the array
def tresholder(phaselist,high_threshold,neighbor_threshold):
    for i in range(1, len(phaselist) - 1):  # Exclude the first and last element
        # If the current value is greater than high_threshold
        # and both neighbors are less than neighbor_threshold
        if (phaselist[i] > high_threshold and
            phaselist[i-1] < neighbor_threshold and
            phaselist[i+1] < neighbor_threshold):
            phaselist[i] = 0  # Set the current value to 0
    return phaselist

phaselist_r = tresholder(phaselist_r,high_threshold,neighbor_threshold)
phaselist_rRef = tresholder(phaselist_rRef,high_threshold,neighbor_threshold)
phaselist_t = tresholder(phaselist_t,high_threshold,neighbor_threshold)
phaselist_tRef = tresholder(phaselist_tRef,high_threshold,neighbor_threshold)

phaselist_r2 = phaselist_r / 180 * np.pi
phaselist_rRef2 = phaselist_rRef / 180 * np.pi
phaselist_t2 = phaselist_t / 180 * np.pi
phaselist_tRef2 = phaselist_tRef / 180 * np.pi

phaselist_p2 = phaselist_p / 180 * np.pi
phaselist_pRef2 = phaselist_pRef / 180 * np.pi


plt.figure(figsize=(12, 4))
plt.plot([(-0.0 + i*15/90) for i in range(len(strlist_r))], strlist_r, '-', color='black', linewidth=1, label='Density Perturbations (PM)')
plt.plot([(-0.0 + i*15/90)  for i in range(len(strlist_t))], strlist_t, '-', color='red', linewidth=1, label='Temperature Perturbations (PM)')
plt.plot([(-0.0 + i*15/90) for i in range(len(strlist_p))], strlist_p, '-', color='blue', linewidth=1, label='Pressure Perturbations (PM)')
plt.plot([(-0.0 + i*15/90)  for i in range(len(strlist_rRef))], strlist_rRef, '--', color='black', linewidth=0.5, label='Density Perturbations (Reference)')
plt.plot([(-0.0 + i*15/90)  for i in range(len(strlist_tRef))], strlist_tRef, '--', color='red', linewidth=0.5, label='Temperature Perturbations (Reference)')
plt.plot([(-0.0 + i*15/90)  for i in range(len(strlist_pRef))], strlist_pRef, '--', color='blue', linewidth=0.5, label='Pressure Perturbations (Reference)')
plt.xlabel('Probe Location (mm)',fontsize=14)
plt.ylabel(r'Perturbation Amplitude (\%)', fontsize=14)

# Add a note in the lower left
plt.annotate('w = 500kHz', xy=(0.043, 0.9), xycoords='axes fraction', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
# Since you called plt.legend() twice, only the second one will take effect. If you want both positions, you need two legend handles
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left')
plt.legend(by_label.values(), by_label.keys(), loc='upper right')
plt.ylim(0, 5)

plt.xticks(np.linspace(0, 14, 15))
plt.grid(axis='x')

plt.show()






fig, (ax2, ax3,ax4) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

ax2.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_p2))], phaselist_p2, '-', color='blue', linewidth=1, label='Pressure')
ax2.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_pRef2))], phaselist_pRef2, '--', color='blue', linewidth=1, label='Temperature_REF')
ax2.set_ylabel('Phase (degrees)')
ax2.legend(loc='upper left')
ax2.grid(axis='x')
ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
# Plot phaselist in the middle subplot
ax3.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_t2))], phaselist_t2, '-', color='red', linewidth=1, label='Temperature')
ax3.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_tRef2))], phaselist_tRef2, '--', color='red', linewidth=1, label='Density_REF')

# # Plot stdDevList in the bottom subplot
# ax3_r = ax3.twinx()
# ax3_r.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_r))], stdDevList_r, '-', color='gray', linewidth=1, label='Density')
# ax3.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_t))], stdDevList_t, '-', color='red', linewidth=1, label='Temperature')
# ax3.plot([(-0.0 + i*15/90) for i in range(len(stdDevList_p))], stdDevList_p, '-', color='blue', linewidth=1, label='Pressure')
# ax3.set_xlabel('Probe Location(mm)')
# ax3.set_ylabel('STD dev')
ax3.set_ylabel('Density', color='gray')
ax3.legend(loc='upper left')
ax3.legend(loc='upper right')
ax3.grid(axis='x')
ax3.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

# Set xticks
# ax3.set_xticks([(-0.0 + i*15/90) for i in range(len(stdDevList_r))])
ax3.set_xticks(np.linspace(0, 14, 15))
# ax1.set_xticks([(-1.5 + i*1.5) for i in range(len(stdDevList_r))])

ax4.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_r2))], phaselist_r2, '-', color='black', linewidth=1, label='Density')
ax4.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_rRef2))], phaselist_rRef2, '--', color='black', linewidth=1, label='Temperature_REF')


# # Plot stdDevList in the bottom subplot
# ax3_r = ax3.twinx()
# ax3_r.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_r))], stdDevList_r, '-', color='gray', linewidth=1, label='Density')
# ax3.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_t))], stdDevList_t, '-', color='red', linewidth=1, label='Temperature')
# ax3.plot([(-0.0 + i*15/90) for i in range(len(stdDevList_p))], stdDevList_p, '-', color='blue', linewidth=1, label='Pressure')
ax4.set_xlabel('Probe Location(mm)')
ax4.set_ylabel('STD dev')
ax4.set_ylabel('Density', color='gray')
ax4.legend(loc='upper left')
ax4.legend(loc='upper right')
ax4.grid(axis='x')
ax4.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

ax4.set_xticks(np.linspace(0, 14, 15))
# ax1.set_xticks([(-1.5 + i*1.5) for i in range(len(stdDevList_r))])
plt.tight_layout()

file_pathplot = f"./PM_feb/{caseName}/plot_{Targetfreq}.png"
plt.savefig(file_pathplot)
plt.show()


plt.figure(figsize=(12, 2))
plt.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_r2))], phaselist_r2, '-', color='black', linewidth=1.5, label='Density perturbations (PM)')
plt.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_rRef2))], phaselist_rRef2, '--', color='black', linewidth=0.75, label='Density perturbations (Reference)')
# # Plot stdDevList in the bottom subplot
# ax3_r = ax3.twinx()
# ax3_r.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_r))], stdDevList_r, '-', color='gray', linewidth=1, label='Density')
# ax3.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_t))], stdDevList_t, '-', color='red', linewidth=1, label='Temperature')
# ax3.plot([(-0.0 + i*15/90) for i in range(len(stdDevList_p))], stdDevList_p, '-', color='blue', linewidth=1, label='Pressure')
plt.xlabel('Probe Location(mm)',fontsize=14)
plt.ylabel('Phase of Density\n Perturbations', fontsize=14, multialignment='center')
# plt.annotate('w = 100kHz', xy=(0.043, 0.9), xycoords='axes fraction', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.legend(loc='upper left')
plt.legend(loc='upper right')
plt.grid(axis='x')
plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.xticks(np.linspace(0, 14, 15))
plt.show()


plt.figure(figsize=(12, 2))
plt.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_t2))], phaselist_t2, '-', color='red', linewidth=1.5, label='Temperature perturbations (PM)')
plt.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_tRef2))], phaselist_tRef2, '--', color='red', linewidth=0.75, label='Temperature perturbations (Reference)')
# # Plot stdDevList in the bottom subplot
# ax3_r = ax3.twinx()
# ax3_r.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_r))], stdDevList_r, '-', color='gray', linewidth=1, label='Density')
# ax3.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_t))], stdDevList_t, '-', color='red', linewidth=1, label='Temperature')
# ax3.plot([(-0.0 + i*15/90) for i in range(len(stdDevList_p))], stdDevList_p, '-', color='blue', linewidth=1, label='Pressure')
plt.xlabel('Probe Location(mm)',fontsize=14)
plt.ylabel('Phase of Temperature\n Perturbations', fontsize=14, multialignment='center')
# plt.annotate('w = 100kHz', xy=(0.043, 0.9), xycoords='axes fraction', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.legend(loc='upper left')
plt.legend(loc='upper right')
plt.grid(axis='x')
plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.xticks(np.linspace(0, 14, 15))
plt.show()




plt.figure(figsize=(12, 2))
plt.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_p2))], phaselist_p2, '-', color='blue', linewidth=1.5, label='Pressure perturbations (PM)')
plt.plot([(-0.0 + i*15/90)  for i in range(len(phaselist_pRef2))], phaselist_pRef2, '--', color='blue', linewidth=0.75, label='Pressure perturbations (Reference)')
# # Plot stdDevList in the bottom subplot
# ax3_r = ax3.twinx()
# ax3_r.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_r))], stdDevList_r, '-', color='gray', linewidth=1, label='Density')
# ax3.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_t))], stdDevList_t, '-', color='red', linewidth=1, label='Temperature')
# ax3.plot([(-0.0 + i*15/90) for i in range(len(stdDevList_p))], stdDevList_p, '-', color='blue', linewidth=1, label='Pressure')
plt.xlabel('Probe Location(mm)',fontsize=14)
plt.ylabel('Phase of Pressure\n Perturbations', fontsize=14, multialignment='center')
# plt.annotate('w = 100kHz', xy=(0.043, 0.9), xycoords='axes fraction', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.legend(loc='upper left')
plt.legend(loc='upper right')
plt.grid(axis='x')
plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.xticks(np.linspace(0, 14, 15))
plt.show()





plt.figure(figsize=(12, 3))
plt.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_r))], stdDevList_r/np.mean(stdDevList_r), '-', color='gray', linewidth=1, label='Density')
plt.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_t))], stdDevList_t/np.mean(stdDevList_t), '-', color='red', linewidth=1, label='Temperature')
plt.plot([(-0.0 + i*15/90) for i in range(len(stdDevList_p))], stdDevList_p/np.mean(stdDevList_p), '-', color='blue', linewidth=1, label='Pressure')
# # Plot stdDevList in the bottom subplot
# ax3_r = ax3.twinx()
# ax3_r.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_r))], stdDevList_r, '-', color='gray', linewidth=1, label='Density')
# ax3.plot([(-0.0 + i*15/90)  for i in range(len(stdDevList_t))], stdDevList_t, '-', color='red', linewidth=1, label='Temperature')
# ax3.plot([(-0.0 + i*15/90) for i in range(len(stdDevList_p))], stdDevList_p, '-', color='blue', linewidth=1, label='Pressure')
plt.xlabel('Probe Location(mm)',fontsize=14)
plt.ylabel('Normalized RMS', fontsize=14, multialignment='center')
# plt.annotate('w = 100kHz', xy=(0.043, 0.9), xycoords='axes fraction', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.legend(loc='upper left')
plt.legend(loc='upper right')
plt.grid(axis='x')


plt.xticks(np.linspace(0, 14, 15))
plt.show()

