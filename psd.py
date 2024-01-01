import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
# Read the file into a DataFrame, assuming space-separated values and no header
caseName = "case2_1mm"
# caseName = "CASE2"
point = 6
# caseName = "case1_2mm"
# caseName = "oscilatingPM50"

file_path = f"./PM_BC_december/{caseName}/Point{str(point)}.dat"
file_path2 = f"./PM_BC_december/ref/Point{str(point)}.dat"
# file_path2 = f"./PM_BC_december/case1_2mm/Point{str(point)}.dat"

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

def plotPSD2(data, data2, probe):
    tau = 5.0e-9   
    fig_xsize = 12
    indexp = 16 #pressure
    # index = 10 #xvel
    # index = 11 #yvel
    indext = 13 #transtemp
    indexrho = 9
    time_step = np.array(data[0]) * tau
    time_step2 = np.array(data2[0]) * tau
    # 10vx 11vy 13t 14trot 15tvib 16p
    data_of_interest = np.array(data[[indexp,indext,indexrho]])
    data_of_interest2 = np.array(data2[[indexp,indext,indexrho]])
    cutoff = len(data_of_interest[:,0])//5
    cutoff2 = len(data_of_interest2[:,0])//5
    # cutoffTime = cutoff * tau
    # cutoffTime2 = cutoff2 * tau
    actual_data = data_of_interest[cutoff:]
    actual_data2 = data_of_interest2[cutoff2:]
    cutoff_timesteps = time_step[cutoff:]
    # cutoff_timesteps2 = time_step2[cutoff2:]
    asd=9*(10**8)*2*(10**12)
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,0],"blue",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('Pressure (pa)')
    plt.title('Pressure vs Time')
    plt.show()
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,1],"red",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time')
    plt.show()
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,2],"black",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('nparticles (#)')
    plt.title('nparticles vs Time')
    plt.show()
    # # # Detrend the data to make it stationary
    detrended_data_P = np.array(signal.detrend(actual_data[:,0]))
    detrended_data2_P = np.array(signal.detrend(actual_data2[:,0]))
    detrended_data_T = np.array(signal.detrend(actual_data[:,1]))
    detrended_data2_T = np.array(signal.detrend(actual_data2[:,1]))
    detrended_data_rho = np.array(signal.detrend(actual_data[:,2]))
    detrended_data2_rho = np.array(signal.detrend(actual_data2[:,2]))
    # Calculate the Power Spectral Density using Fast Fourier Transform
    samplingFreq = 1/tau
    frequencies, psd_valuesP = signal.welch(detrended_data_P, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2P = signal.welch(detrended_data2_P, fs=samplingFreq, nperseg=100000)
    frequencies, psd_valuesT = signal.welch(detrended_data_T, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2T = signal.welch(detrended_data2_T, fs=samplingFreq, nperseg=100000)
    frequencies, psd_valuesRho = signal.welch(detrended_data_rho, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2Rho = signal.welch(detrended_data2_rho, fs=samplingFreq, nperseg=100000)
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesP,"orange",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2P,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Pressure PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [P**2/Hz]', fontsize=18)
    plt.xlim([2*10**1, 2*10**5])
    # plt.xlim(left=2*10**1)
    plt.ylim([2*10**-10, 2*10**-5]) # setting y-axis range
    # plt.ylim(bottom=2*10**-8)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesT,"red",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2T,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Translational Temperature PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [T**2/Hz]', fontsize=18)
    plt.xlim([2*10**1, 2*10**5])
    # plt.ylim(bottom=2*10**-7)
    # plt.xlim(left=2*10**1)
    plt.ylim([2*10**-10, 2*10**-4]) # setting y-axis range
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesRho,"black",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2Rho,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Density PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [rho**2/Hz]', fontsize=18)
    plt.xlim([2*10**1, 2*10**5])
    # plt.ylim(bottom=2*10**-8)
    # plt.xlim(left=2*10**1)
    plt.ylim([10**-10, 5*10**-3]) # setting y-axis range
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
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

def plotPSD2_kucuk(data, data2, probe):
    xxmin = 2*10**1
    xxmax = 10**5
    tau = 5.0e-9   
    fig_xsize = 8
    indexp = 16 #pressure
    # index = 10 #xvel
    # index = 11 #yvel
    indext = 13 #transtemp
    indexrho = 9
    time_step = np.array(data[0]) * tau
    time_step2 = np.array(data2[0]) * tau
    # 10vx 11vy 13t 14trot 15tvib 16p
    data_of_interest = np.array(data[[indexp,indext,indexrho]])
    data_of_interest2 = np.array(data2[[indexp,indext,indexrho]])
    cutoff = len(data_of_interest[:,0])//5
    cutoff2 = len(data_of_interest2[:,0])//5
    # cutoffTime = cutoff * tau
    # cutoffTime2 = cutoff2 * tau
    actual_data = data_of_interest[cutoff:]
    actual_data2 = data_of_interest2[cutoff2:]
    cutoff_timesteps = time_step[cutoff:]
    # cutoff_timesteps2 = time_step2[cutoff2:]
    asd=9*(10**8)*2*(10**12)
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,0],"blue",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('Pressure (pa)')
    plt.title('Pressure vs Time')
    plt.show()
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,1],"red",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time')
    plt.show()
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,2],"black",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('nparticles (#)')
    plt.title('nparticles vs Time')
    plt.show()
    # # # Detrend the data to make it stationary
    detrended_data_P = np.array(signal.detrend(actual_data[:,0]))
    detrended_data2_P = np.array(signal.detrend(actual_data2[:,0]))
    detrended_data_T = np.array(signal.detrend(actual_data[:,1]))
    detrended_data2_T = np.array(signal.detrend(actual_data2[:,1]))
    detrended_data_rho = np.array(signal.detrend(actual_data[:,2]))
    detrended_data2_rho = np.array(signal.detrend(actual_data2[:,2]))
    # Calculate the Power Spectral Density using Fast Fourier Transform
    samplingFreq = 1/tau
    frequencies, psd_valuesP = signal.welch(detrended_data_P, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2P = signal.welch(detrended_data2_P, fs=samplingFreq, nperseg=100000)
    frequencies, psd_valuesT = signal.welch(detrended_data_T, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2T = signal.welch(detrended_data2_T, fs=samplingFreq, nperseg=100000)
    frequencies, psd_valuesRho = signal.welch(detrended_data_rho, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2Rho = signal.welch(detrended_data2_rho, fs=samplingFreq, nperseg=100000)
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesP,"orange",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2P,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Pressure PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [P**2/Hz]', fontsize=18)
    plt.xlim([xxmin, xxmax])
    # plt.xlim(left=2*10**1)
    plt.ylim([2*10**-10, 2*10**-5]) # setting y-axis range
    # plt.ylim(bottom=2*10**-8)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesT,"red",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2T,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Translational Temperature PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [T**2/Hz]', fontsize=18)
    plt.xlim([xxmin, xxmax])
    # plt.ylim(bottom=2*10**-7)
    # plt.xlim(left=2*10**1)
    plt.ylim([2*10**-10, 2*10**-4]) # setting y-axis range
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesRho,"black",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2Rho,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Density PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [rho**2/Hz]', fontsize=18)
    plt.xlim([xxmin, xxmax])
    # plt.ylim(bottom=2*10**-8)
    # plt.xlim(left=2*10**1)
    plt.ylim([10**-10, 5*10**-3]) # setting y-axis range
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

def plotPSD2_100(data, data2, probe):
    tau = 5.0e-9   
    fig_xsize = 12
    indexp = 16 #pressure
    # index = 10 #xvel
    # index = 11 #yvel
    indext = 13 #transtemp
    indexrho = 9
    time_step = np.array(data[0]) * tau
    time_step2 = np.array(data2[0]) * tau
    # 10vx 11vy 13t 14trot 15tvib 16p
    data_of_interest = np.array(data[[indexp,indext,indexrho]])
    data_of_interest2 = np.array(data2[[indexp,indext,indexrho]])
    cutoff = len(data_of_interest[:,0])//5
    cutoff2 = len(data_of_interest2[:,0])//5
    # cutoffTime = cutoff * tau
    # cutoffTime2 = cutoff2 * tau
    actual_data = data_of_interest[cutoff:]
    actual_data2 = data_of_interest2[cutoff2:]
    cutoff_timesteps = time_step[cutoff:]
    # cutoff_timesteps2 = time_step2[cutoff2:]
    asd=9*(10**8)*2*(10**12)
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,0],"blue",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('Pressure (pa)')
    plt.title('Pressure vs Time')
    plt.show()
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,1],"red",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time')
    plt.show()
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,2],"black",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('nparticles (#)')
    plt.title('nparticles vs Time')
    plt.show()
    # # # Detrend the data to make it stationary
    detrended_data_P = np.array(signal.detrend(actual_data[:,0]))
    detrended_data2_P = np.array(signal.detrend(actual_data2[:,0]))
    detrended_data_T = np.array(signal.detrend(actual_data[:,1]))
    detrended_data2_T = np.array(signal.detrend(actual_data2[:,1]))
    detrended_data_rho = np.array(signal.detrend(actual_data[:,2]))
    detrended_data2_rho = np.array(signal.detrend(actual_data2[:,2]))
    # Calculate the Power Spectral Density using Fast Fourier Transform
    samplingFreq = 1/tau
    frequencies, psd_valuesP = signal.welch(detrended_data_P, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2P = signal.welch(detrended_data2_P, fs=samplingFreq, nperseg=100000)
    frequencies, psd_valuesT = signal.welch(detrended_data_T, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2T = signal.welch(detrended_data2_T, fs=samplingFreq, nperseg=100000)
    frequencies, psd_valuesRho = signal.welch(detrended_data_rho, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2Rho = signal.welch(detrended_data2_rho, fs=samplingFreq, nperseg=100000)
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesP,"orange",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2P,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Pressure PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [P**2/Hz]', fontsize=18)
    plt.xlim([2*10**1, 2*10**5])
    # plt.xlim(left=2*10**1)
    plt.ylim([2*10**-10, 2*10**-4]) # setting y-axis range
    # plt.ylim(bottom=2*10**-8)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesT,"red",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2T,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Translational Temperature PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [T**2/Hz]', fontsize=18)
    plt.xlim([2*10**1, 2*10**5])
    # plt.ylim(bottom=2*10**-7)
    # plt.xlim(left=2*10**1)
    plt.ylim([2*10**-10, 2*10**-3]) # setting y-axis range
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesRho,"black",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2Rho,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Density PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [rho**2/Hz]', fontsize=18)
    plt.xlim([2*10**1, 2*10**5])
    # plt.ylim(bottom=2*10**-8)
    # plt.xlim(left=2*10**1)
    plt.ylim([10**-10, 5*10**-2]) # setting y-axis range
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


def plotPSD2_jetcomp(data, data2, probe):
    tau = 5.0e-9   
    fig_xsize = 12
    indexp = 16 #pressure
    # index = 10 #xvel
    # index = 11 #yvel
    indext = 13 #transtemp
    indexrho = 9
    time_step = np.array(data[0]) * tau
    time_step2 = np.array(data2[0]) * tau
    # 10vx 11vy 13t 14trot 15tvib 16p
    data_of_interest = np.array(data[[indexp,indext,indexrho]])
    data_of_interest2 = np.array(data2[[indexp,indext,indexrho]])
    cutoff = len(data_of_interest[:,0])//5
    cutoff2 = len(data_of_interest2[:,0])//5
    # cutoffTime = cutoff * tau
    # cutoffTime2 = cutoff2 * tau
    actual_data = data_of_interest[cutoff:]
    actual_data2 = data_of_interest2[cutoff2:]
    cutoff_timesteps = time_step[cutoff:]
    # cutoff_timesteps2 = time_step2[cutoff2:]
    asd=9*(10**8)*2*(10**12)
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,0],"blue",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('Pressure (pa)')
    plt.title('Pressure vs Time')
    plt.show()
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,1],"red",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time')
    plt.show()
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(cutoff_timesteps, actual_data[:,2],"black",linewidth=0.3, linestyle='-')
    plt.xlabel('Time (s))')
    plt.ylabel('nparticles (#)')
    plt.title('nparticles vs Time')
    plt.show()
    # # # Detrend the data to make it stationary
    detrended_data_P = np.array(signal.detrend(actual_data[:,0]))
    detrended_data2_P = np.array(signal.detrend(actual_data2[:,0]))
    detrended_data_T = np.array(signal.detrend(actual_data[:,1]))
    detrended_data2_T = np.array(signal.detrend(actual_data2[:,1]))
    detrended_data_rho = np.array(signal.detrend(actual_data[:,2]))
    detrended_data2_rho = np.array(signal.detrend(actual_data2[:,2]))
    # Calculate the Power Spectral Density using Fast Fourier Transform
    samplingFreq = 1/tau
    frequencies, psd_valuesP = signal.welch(detrended_data_P, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2P = signal.welch(detrended_data2_P, fs=samplingFreq, nperseg=100000)
    frequencies, psd_valuesT = signal.welch(detrended_data_T, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2T = signal.welch(detrended_data2_T, fs=samplingFreq, nperseg=100000)
    frequencies, psd_valuesRho = signal.welch(detrended_data_rho, fs=samplingFreq, nperseg=100000)
    frequencies2, psd_values2Rho = signal.welch(detrended_data2_rho, fs=samplingFreq, nperseg=100000)
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesP,"orange",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2P,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Pressure PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [P**2/Hz]', fontsize=18)
    plt.xlim([2*10**1, 2*10**5])
    # plt.xlim(left=2*10**1)
    plt.ylim([2*10**-10, 2*10**-4]) # setting y-axis range
    # plt.ylim(bottom=2*10**-8)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesT,"red",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2T,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Translational Temperature PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [T**2/Hz]', fontsize=18)
    plt.xlim([2*10**1, 2*10**5])
    # plt.ylim(bottom=2*10**-7)
    # plt.xlim(left=2*10**1)
    plt.ylim([2*10**-10, 2*10**-3]) # setting y-axis range
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    # Plotting the Power Spectral Density
    plt.figure(figsize=(fig_xsize, 6))
    plt.loglog(frequencies/1000, psd_valuesRho,"black",linewidth=3, label=f'probe {probe}') #label='Perturbed Flow via Pulsed Jet'
    plt.loglog(frequencies2/1000, psd_values2Rho,linewidth=1, label='reference') #label='No Jet(Reference State)'
    plt.title('Density PSD', fontsize=22)
    plt.xlabel('Frequency [kHz]', fontsize=18)
    plt.ylabel('PSD [rho**2/Hz]', fontsize=18)
    plt.xlim([2*10**1, 2*10**5])
    # plt.ylim(bottom=2*10**-8)
    # plt.xlim(left=2*10**1)
    plt.ylim([10**-10, 5*10**-2]) # setting y-axis range
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
# plotPSD(df)
plotPSD2(df,df2, point)
# plotPSD2_100(df,df2, point)
# plotPSD2_kucuk(df,df2, point)
# plotPSD2_jetcomp(df,df2, point)