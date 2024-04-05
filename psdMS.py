import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

# tau = 5.0e-9 

# # Read the file into a DataFrame, assuming space-separated values and no header
# caseName = "100_15at4_sync"
# # caseName = "CASE2"
# probe = 40
  

# file_path = f"./PM_april/{caseName}/PROBE_{str(point)}"
# file_path2 = f"./PM_april/ref_edge/PROBE_{str(point+10)}"
# # file_path2 = f"./PM_BC_december/case1_2mm/Point{str(point)}.dat"

# data = pd.read_csv(file_path, delim_whitespace=True, header=None)
# data2 = pd.read_csv(file_path2, delim_whitespace=True, header=None)

# fig_xsize = 12
# indexp = 16 #pressure
# # index = 10 #xvel
# # index = 11 #yvel
# indext = 13 #transtemp
# indexrho = 9
# time_step = np.array(data[0]) * tau
# time_step2 = np.array(data2[0]) * tau
# data_of_interest = np.array(data[[indext]])
# data_of_interest2 = np.array(data2[[indext]])

# cutoff = len(data_of_interest[:,0])//5
# cutoff2 = len(data_of_interest2[:,0])//5
# actual_data = data_of_interest[cutoff:]
# actual_data2 = data_of_interest2[cutoff2:]
# cutoff_timesteps = time_step[cutoff:]


# detrended_data_T = np.array(signal.detrend(actual_data[:,0]))
# detrended_data2_T = np.array(signal.detrend(actual_data2[:,0]))

# samplingFreq = 1/tau

# frequencies, psd_valuesT = signal.welch(detrended_data_T, fs=samplingFreq, nperseg=100000)
# frequencies2, psd_values2T = signal.welch(detrended_data2_T, fs=samplingFreq, nperseg=100000)

# plt.figure(figsize=(fig_xsize, 6))
# plt.loglog(frequencies/1000, psd_valuesT,"red",linewidth=1, label=f'probe {probe} before Shock') #label='Perturbed Flow via Pulsed Jet'
# plt.loglog(frequencies2/1000, psd_values2T,linewidth=1, label=f'probe {probe+10} after Shock') #label='No Jet(Reference State)'
# plt.title('Translational Temperature PSD', fontsize=22)
# plt.xlabel('Frequency [kHz]', fontsize=18)
# plt.ylabel('PSD [T**2/Hz]', fontsize=18)
# plt.xlim([2*10**1, 2*10**4])
# plt.ylim([5*10**-9, 2*10**-3]) # setting y-axis range
# plt.legend(fontsize=20)
# plt.grid(True)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.show()

# Initialize constants


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def plot_all_probes_psd(start_probe, end_probe, case_name, ref_name, index_of_interest, tau=5.0e-9):
    fig_xsize = 12
    sampling_freq = 1/tau
    nperseg = 100000  # The segment length for the Welch method
    
    # Define the frequency range for the y-axis in Hz (convert to kHz if necessary)
    freq_min = 1e3  # 50 kHz
    freq_max = 1000e3  # 500 kHz
    
    # Initialize lists to store frequency and PSD data
    all_frequencies = []
    all_psd_values = []
    all_frequencies2 = []
    all_psd_values2 = []
    
    # Loop over each probe
    for point in range(start_probe, end_probe + 1):
        file_path = f"./PM_april/{case_name}/PROBE_{str(point)}"
        file_path2 = f"./PM_april/{ref_name}/PROBE_{str(point)}"
        # Read the data
        data = pd.read_csv(file_path, delim_whitespace=True, header=None)
        data2 = pd.read_csv(file_path2, delim_whitespace=True, header=None)

        data_of_interest = np.array(data[index_of_interest])
        data_of_interest2 = np.array(data2[index_of_interest])

        # Detrend and calculate PSD
        asdasd = len(data_of_interest)//5
        asdasd = 40000

        detrended_data = signal.detrend(data_of_interest[asdasd:])
        frequencies, psd_values = signal.welch(detrended_data, fs=sampling_freq, nperseg=nperseg)
        
        detrended_data2 = signal.detrend(data_of_interest2[asdasd:])
        frequencies2, psd_values2 = signal.welch(detrended_data2, fs=sampling_freq, nperseg=nperseg)

        # Filter the frequencies and PSD values to match the desired frequency range
        valid_freq_indices = (frequencies >= freq_min) & (frequencies <= freq_max)
        frequencies = frequencies[valid_freq_indices]
        psd_values = psd_values[valid_freq_indices]
        # Store the filtered results
        all_frequencies.append(frequencies)
        all_psd_values.append(psd_values)

        valid_freq_indices2 = (frequencies2 >= freq_min) & (frequencies2 <= freq_max)
        frequencies2 = frequencies2[valid_freq_indices2]
        psd_values2 = psd_values2[valid_freq_indices2]
        # Store the filtered results
        all_frequencies2.append(frequencies2)
        all_psd_values2.append(psd_values2)

    
    # Assuming all frequency arrays are the same after filtering (which they should be)
    X, Y = np.meshgrid(range(start_probe, end_probe + 1), all_frequencies[0])
    
    # Prepare the PSD values for contour plotting
    Z = np.array(all_psd_values)
    vmin, vmax =-8, -2

    plt.figure(figsize=(fig_xsize, 6))
    # contour = plt.contourf(X, Y/1000, 10*np.log10(Z.T), levels=100, cmap='viridis')  # Convert frequency to kHz
    contour = plt.contourf(X, Y/1000, (Z.T), levels=100, cmap='viridis')  # Convert frequency to kHz
    # contour = plt.contourf(X, Y/1000, 10*np.log10(Z.T), levels=100, cmap='viridis', vmin=vmin, vmax=vmax)  # Convert frequency to kHz
    plt.colorbar(contour)
    # Set the y-axis limits to the desired frequency range
    plt.ylim(freq_min/1000, freq_max/1000)  # Convert frequency to kHz for the plot
    plt.xlabel('Probe Number')
    plt.ylabel('Frequency [kHz]')  # Make sure to label the units correctly as kHz
    plt.title(f'PSD - {case_name} across Probes {start_probe} to {end_probe}')
    plt.show()


    # Plot
    plt.figure(figsize=(fig_xsize, 6))
    # contour = plt.contourf(X, Y/1000, 10*np.log10(Z.T), levels=100, cmap='viridis')  # Convert frequency to kHz
    contour = plt.contourf(X, Y/1000, np.log10(Z.T), levels=100, cmap='viridis', vmin=vmin, vmax=vmax)  # Convert frequency to kHz
    # contour = plt.contourf(X, Y/1000, 10*np.log10(Z.T), levels=100, cmap='viridis', vmin=vmin, vmax=vmax)  # Convert frequency to kHz
    plt.colorbar(contour)
    # Set the y-axis limits to the desired frequency range
    plt.ylim(freq_min/1000, freq_max/1000)  # Convert frequency to kHz for the plot
    plt.xlabel('Probe Number')
    plt.ylabel('Frequency [kHz]')  # Make sure to label the units correctly as kHz
    plt.title(f'PSD - {case_name} across Probes {start_probe} to {end_probe}')
    plt.show()

    X2, Y2 = np.meshgrid(range(start_probe, end_probe + 1), all_frequencies2[0])
    # Prepare the PSD values for contour plotting
    Z2 = np.array(all_psd_values2)
    # Plot
    plt.figure(figsize=(fig_xsize, 6))
    contour = plt.contourf(X2, Y2/1000, (Z2.T), levels=100, cmap='viridis')  # Convert frequency to kHz
    plt.colorbar(contour)
    # Set the y-axis limits to the desired frequency range
    plt.ylim(freq_min/1000, freq_max/1000)  # Convert frequency to kHz for the plot
    plt.xlabel('Probe Number')
    plt.ylabel('Frequency [kHz]')  # Make sure to label the units correctly as kHz
    plt.title(f'PSD - {case_name} across Probes {start_probe} to {end_probe}')
    plt.show()
    #+=============================================================================================
    plt.figure(figsize=(fig_xsize, 6))
    contour = plt.contourf(X2, Y2/1000, np.log10(Z2.T), levels=100, cmap='viridis',vmin=vmin, vmax=vmax)  # Convert frequency to kHz
    plt.colorbar(contour)
    # Set the y-axis limits to the desired frequency range
    plt.ylim(freq_min/1000, freq_max/1000)  # Convert frequency to kHz for the plot
    plt.xlabel('Probe Number')
    plt.ylabel('Frequency [kHz]')  # Make sure to label the units correctly as kHz
    plt.title(f'PSD - {case_name} across Probes {start_probe} to {end_probe}')
    plt.show()
    return 



# Call the function with the specified parameters
plot_all_probes_psd(start_probe=1, end_probe=98, case_name="100khzat1_30at11_HUGE",ref_name="refHuge", index_of_interest=13)

# # Use the function for your case
# plot_all_probes_psd(start_probe=1, end_probe=90, case_name="100_15at4_sync", index_of_interest=indext)
