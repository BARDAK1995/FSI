import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
# Read the file into a DataFrame, assuming space-separated values and no header
caseName = "CASE1_half"
# caseName = "CASE2"
point = 3
caseName = "oscVel_pm"
# caseName = "oscilatingPM50"

file_path = f"./PM_BC_1/{caseName}/Point{str(point)}.dat"
file_path2 = f"./PM_BC_1/ref/Point{str(point)}.dat"

# file_path2 = f"./PM_BC_1/{caseName}/Point{str(point+5)}.dat"
# file_path2 = f"./Phonic_state_MS3/{caseName}/Point{str(point)}.dat"
# file_path = "8Hugejet.dat"

df = pd.read_csv(file_path, delim_whitespace=True, header=None)
df2 = pd.read_csv(file_path2, delim_whitespace=True, header=None)
# Extract the first column (time step) and the 10th column (data of interest)

def plotPSD(data):
    tau = 5.0e-9   
    time_step = np.array(data[0]) * tau
    data_of_interest = np.array(data[13])
    cutoff = len(data_of_interest)//2
    cutoffTime = cutoff * tau
    actual_data = data_of_interest[cutoff:]
    cutoff_timesteps = time_step[cutoff:]

    # # Plot the data
    # plt.figure(figsize=(12, 6))
    # plt.plot(cutoff_timesteps, actual_data, marker='o', linestyle='-')
    # plt.xlabel('Time (s))')
    # plt.ylabel('Pressure (Pa)')
    # plt.title('Pressure vs Time')
    # # plt.grid(True)
    # plt.show()

    # # # Detrend the data to make it stationary
    detrended_data = signal.detrend(actual_data)
    # plt.figure(figsize=(12, 6))
    # plt.plot(cutoff_timesteps, detrended_data, marker='o', linestyle='-')
    # plt.xlabel('Time (s))')
    # plt.ylabel('Pressure Perturbation (Pa)')
    # plt.title('Unsteady Pressure')
    # # plt.grid(True)
    # plt.show()
    detrended_data_array = np.array(detrended_data)
    # Calculate the Power Spectral Density using Fast Fourier Transform
    samplingFreq = 1/tau
    frequencies, psd_values = signal.welch(detrended_data_array, fs=samplingFreq, nperseg=10030)
    # Plotting the Power Spectral Density
    plt.figure(figsize=(12, 6))
    plt.loglog(frequencies/1000, psd_values)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.ylim([10**-12, 1*10**1]) # setting y-axis range

    plt.grid(True)
    plt.show()

def plotPSD2(data, data2):
    tau = 5.0e-9   
    index = 16 #pressure
    # index = 10 #xvel
    # index = 11 #yvel
    # index = 13 #transtemp
    # index = 9
    time_step = np.array(data[0]) * tau
    time_step2 = np.array(data2[0]) * tau
    # 10vx 11vy 13t 14trot 15tvib 16p
    data_of_interest = np.array(data[index])
    data_of_interest2 = np.array(data2[index])

    cutoff = len(data_of_interest)//6
    cutoff2 = len(data_of_interest2)//6

    cutoffTime = cutoff * tau
    cutoffTime2 = cutoff2 * tau
    
    actual_data = data_of_interest[cutoff:]
    actual_data2 = data_of_interest2[cutoff2:]

    cutoff_timesteps = time_step[cutoff:]
    cutoff_timesteps2 = time_step2[cutoff2:]
    asd=9*(10**8)*2*(10**12)
    # Plot the data
    plt.figure(figsize=(12, 6))
    # plt.plot(cutoff_timesteps, actual_data, marker='o', linestyle='-')
    plt.plot(cutoff_timesteps, actual_data,"blue",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('Pressure (pa)')
    plt.title('Pressure vs Time')
    # plt.grid(True)
    plt.show()

    # # # Detrend the data to make it stationary
    detrended_data = signal.detrend(actual_data)
    detrended_data2 = signal.detrend(actual_data2)
    # plt.figure(figsize=(12, 6))
    # plt.plot(cutoff_timesteps, detrended_data, marker='o', linestyle='-')
    # plt.xlabel('Time (s))')
    # plt.ylabel('Pressure Perturbation (Pa)')
    # plt.title('Unsteady Pressure')
    # # plt.grid(True)
    # plt.show()
    detrended_data_array = np.array(detrended_data)
    detrended_data_array2 = np.array(detrended_data2)
    # Calculate the Power Spectral Density using Fast Fourier Transform
    samplingFreq = 1/tau

    frequencies, psd_values = signal.welch(detrended_data_array, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2 = signal.welch(detrended_data_array2, fs=samplingFreq, nperseg=100000)

    # Plotting the Power Spectral Density
    plt.figure(figsize=(12, 6))
    plt.loglog(frequencies/1000, psd_values,"black",linewidth=3, label='probe 3') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2,linewidth=3, label='reference') #label='No Jet(Reference State)'
    plt.title('PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [T**2/Hz]', fontsize=18)
    plt.xlim([5*10**1, 10**4])
    plt.ylim([2*10**-8, 1*10**-1]) # setting y-axis range
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

def plotPSD22(data, data2):
    tau = 5.0e-9   
    time_step = np.array(data[0]) * tau
    time_step2 = np.array(data2[0]) * tau
    # 10vx 11vy 13t 14trot 15tvib 16p
    data_of_interest = np.array(data[16])
    data_of_interest2 = np.array(data2[16])

    cutoff = len(data_of_interest)//4
    cutoff2 = len(data_of_interest2)//4

    cutoffTime = cutoff * tau
    cutoffTime2 = cutoff2 * tau
    
    actual_data = data_of_interest[cutoff:]
    actual_data2 = data_of_interest2[cutoff2:]

    cutoff_timesteps = time_step[cutoff:]
    cutoff_timesteps2 = time_step2[cutoff2:]

    # # Plot the data
    # plt.figure(figsize=(12, 6))
    # plt.plot(cutoff_timesteps, actual_data, marker='o', linestyle='-')
    # plt.xlabel('Time (s))')
    # plt.ylabel('Pressure (Pa)')
    # plt.title('Pressure vs Time')
    # # plt.grid(True)
    # plt.show()

    # # # Detrend the data to make it stationary
    detrended_data = signal.detrend(actual_data)
    detrended_data2 = signal.detrend(actual_data2)
    # plt.figure(figsize=(12, 6))
    # plt.plot(cutoff_timesteps, detrended_data, marker='o', linestyle='-')
    # plt.xlabel('Time (s))')
    # plt.ylabel('Pressure Perturbation (Pa)')
    # plt.title('Unsteady Pressure')
    # # plt.grid(True)
    # plt.show()
    detrended_data_array = np.array(detrended_data)
    detrended_data_array2 = np.array(detrended_data2)
    # Calculate the Power Spectral Density using Fast Fourier Transform
    samplingFreq = 1/tau

    frequencies, psd_values = signal.welch(detrended_data_array, fs=samplingFreq, nperseg=20000)
    frequencies2, psd_values2 = signal.welch(detrended_data_array2, fs=samplingFreq, nperseg=20000)

    # Plotting the Power Spectral Density
    plt.figure(figsize=(12, 6))
    plt.loglog(frequencies/1000, psd_values,linewidth=2.5, label='Case 1')
    plt.loglog(frequencies2/1000, psd_values2, label='Case 2 1/3rho 3xSpeed')
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('PSD [P**2/Hz]')
    plt.ylim([10**-10, 2*10**-2]) # setting y-axis range
    plt.legend()
    plt.grid(True)
    plt.show()
# plotPSD(df)
plotPSD2(df,df2)