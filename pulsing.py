import numpy as np
import matplotlib.pyplot as plt

# Enabling LaTeX formatting in matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Computer Modern Roman"],
})
frequency = 1  # Frequency in Hz
phase_shift = np.pi / 4  # Phase shift in radians
time_period = 1 / frequency  # Time period of one cycle
sampling_rate = 100  # Sampling rate in samples per second
time = np.linspace(0, 2.5 * time_period, int(2.5 * time_period * sampling_rate), endpoint=False)  # Time vector for 3 periods

# Generating the signals
signal_1 = np.sin(2 * np.pi * frequency * time)
signal_2 = np.sin(2 * np.pi * frequency * time - phase_shift)
signal_3 = np.sin(2 * np.pi * frequency * time - 2 * phase_shift)
# Given parameters and signals are already defined in the previous code.
# Recalculate Umin, Umean, and Umax based on new phase_shift
Umax = max(np.max(signal_1), np.max(signal_2), np.max(signal_3))
Umin = min(np.min(signal_1), np.min(signal_2), np.min(signal_3))
Umean = (Umax + Umin) / 2

# Creating the plot with LaTeX formatted labels
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
fig, (ax1) = plt.subplots(1, 1, figsize=(12, 2.5))
# First subplot
ax1.plot(time, signal_1, label=r"$at\ t_0=t_0$", color="black", linewidth=2)
ax1.plot(time, signal_2, label=r"$at\ t_1=t_0+\Delta t$", color="darkblue", linestyle=":", linewidth=1.5)
ax1.plot(time, signal_3, label=r"$at\ t_2=t_0+2\Delta t$", color="darkgreen", linestyle=(0, (3, 5, 1, 5)), linewidth=1.2)
ax1.axvline(x=0.5, color='red', linewidth=0.35)
ax1.axvline(x=0.25, color='red', linewidth=0.35)
ax1.text(0.25, Umin-0.32, 'Probe1', color='red', ha='center', va='bottom',fontsize=15)
ax1.text(0.5, Umin-0.32, 'Probe2', color='red', ha='center', va='bottom',fontsize=15)
ax1.set_yticks([Umin, Umean, Umax])
ax1.set_yticklabels([r'$U_{min}$', r'$U_{mean}$', r'$U_{max}$'])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xticks([])
ax1.set_xlabel("Streamwise Position", fontsize=15)
ax1.set_ylabel("Flow Property U", fontsize=15)
ax1.legend()
ax1.set_xlim(left=0)
ax3.tick_params(axis='both', which='major', labelsize=14)
plt.show()


fig, (ax2,ax3) = plt.subplots(2, 1, figsize=(12, 3))
# First subplot
ax2.plot(time, (time-0.25)%time_period, label=r"$T_0$", color="darkred", linewidth=2)
ax2.axvline(x=0.25, color='red', linewidth=0.35)
ax2.axvline(x=0.5, color='red', linewidth=0.35)
ax2.text(0.25, -0.25, 'Probe1', color='red', ha='center', va='bottom',fontsize=15)
ax2.text(0.5, -0.25, 'Probe2', color='red', ha='center', va='bottom',fontsize=15)
ax2.set_yticks([0, 0.5, 1])
ax2.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xticks([])
ax2.set_xlabel("Streamwise Position", fontsize=15)
ax2.set_ylabel("Phase of\n Perturbations", fontsize=15)
ax2.set_xlim(left=0)
ax2.tick_params(axis='both', which='major', labelsize=14)
# # Second subplot
ax3.plot(time, (time-0.25)%time_period*0, label=r"$T_0$", color="blue", linewidth=2)

ax3.axvline(x=0.25, color='red', linewidth=0.35)
ax3.axvline(x=0.5, color='red', linewidth=0.35)
ax3.text(0.25, -0.077, 'Probe1', color='red', ha='center', va='bottom',fontsize=14)

ax3.text(0.5, -0.077, 'Probe2', color='red', ha='center', va='bottom',fontsize=14)
ax3.set_yticks([0])
ax3.set_yticklabels([r'$A$'])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xticks([])
ax3.set_xlabel("Streamwise Position", fontsize=15)
ax3.set_ylabel("Perturbation\n Amplitude", fontsize=15)
ax3.set_xlim(left=0)
ax3.tick_params(axis='both', which='major', labelsize=14)
plt.show()