import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

amuKg = 1.66e-27
dict_elements = {}
dict_elements["he"] = 4.0026 * amuKg
dict_elements["N2"] = 2*14 * amuKg

def maxwellian_densityWall(v, m, T, k=1.38064852e-23):
    # constant_factor = ((m / (2 * np.pi * k * T)) ** 1.5) * 4 * np.pi * v * v
    constant_factor = (1/2) * (m / ( k * T))**2  * (v**3) 
    exponential_factor = np.exp(-m * (v**2) / (2 * k * T))
    mean_velocity = np.sqrt((9/8) * np.pi * k * T / m )
    mostProbSpeed = 1.41 * (1.38064852e-23 * T / m) ** 0.5
    print(f"Mean speed is {mean_velocity}")
    print(f"most prob speed is {mostProbSpeed}")
    return constant_factor * exponential_factor, mean_velocity


def maxwellian_probabilityFunc_NORMALVEL(v, m, T, k=1.38064852e-23):
    # constant_factor = ((m / (2 * np.pi * k * T)) ** 1.5) * 4 * np.pi * v * v
    constant_factor = (m / ( k * T)) * v
    exponential_factor = np.exp(-m * (v**2) / (2 * k * T))
    mostProbSpeed = 1.41 * (1.38064852e-23 * T / m) ** 0.5
    print(f"most common speed is {mostProbSpeed}")
    return constant_factor * exponential_factor, mostProbSpeed

T = 300 
m = dict_elements["N2"]
unsteadyV = 34.56

v_values = np.linspace(0, 1400, 1000)  # Generates 1000 points between 0 and 3000
f_values, mean_Velocity = maxwellian_densityWall(v_values, m, T)
f_values, mean_Velocity = maxwellian_probabilityFunc_NORMALVEL(v_values, m, T)

plt.figure(figsize=(12, 6))

plt.plot(v_values, f_values, label=f'Nitrogen Gas, T = {T}K', linewidth=3, color='black')  # Increased linewidth
# plt.plot(v_values - unsteadyV, f_values, label=f'v - unsteadyV, m={m}, T={T}', linewidth=2)
# plt.plot(v_values + unsteadyV, f_values, label=f'v + unsteadyV, m={m}, T={T}', linewidth=2)

plt.xlabel('Reflected Normal velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Normal Velocity Probability Distribution', fontsize=20)  # Increased fontsize

# Adding an annotation inside the plot
plt.annotate(f'Most Probable Velocity = {mean_Velocity:.1f} m/s', 
             xy=(max(v_values), max(f_values)), 
             xytext=(-10, -50), 
             textcoords='offset points',
             horizontalalignment='right',
             verticalalignment='top',
             fontsize=16, 
             color='Black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))  # Box added here


plt.legend(fontsize=14)  # Increased fontsize
plt.grid(True)  
plt.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # y-axis in scientific notation
plt.xlim(0, 1400)
plt.ylim(0,0.0022)
plt.show()
# plt.tight_layout()

plt.show()




plt.figure(figsize=(12, 6))
plt.plot(v_values, f_values, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
plt.plot(v_values - unsteadyV, f_values, label='Vmin', linewidth=3, color='red')
plt.plot(v_values + unsteadyV, f_values, label='Vmax', linewidth=3, color='blue')

plt.xlabel('Reflected Normal velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Normal Velocity Probability Distribution', fontsize=20)  # Increased fontsize

# Adding an annotation inside the plot
plt.annotate(f'Most Probable Velocity = {mean_Velocity:.1f} ± {unsteadyV}m/s', 
             xy=(max(v_values), max(f_values)), 
             xytext=(-10, -60), 
             textcoords='offset points',
             horizontalalignment='right',
             verticalalignment='top',
             fontsize=16, 
             color='Black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))  # Box added here


plt.legend(fontsize=14)  # Increased fontsize
plt.grid(True)  
plt.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # y-axis in scientific notation
plt.xlim(0, 1400)
plt.ylim(0,0.0022)
plt.show()
# plt.tight_layout()
plt.show()

file_path1 = 'PM_BC2\ANAN1_0.fdf'
file_path2 = 'PM_BC2\ANAN2_0.fdf'
df1 = pd.read_csv(file_path1, delim_whitespace=True, header=None)
df2 = pd.read_csv(file_path2, delim_whitespace=True, header=None)
# Extract the 3rd column (index 2)
third_column_data1 = df1[2]
third_column_data2 = df2[2]
# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(third_column_data1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
plt.hist(third_column_data2, bins=50,density=True,color='red', edgecolor='black',alpha=0.5, label='Vmin')
plt.plot(v_values, f_values, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
plt.plot(v_values - unsteadyV, f_values, label='Vmin', linewidth=3, color='red')
plt.plot(v_values + unsteadyV, f_values, label='Vmax', linewidth=3, color='blue')
plt.xlabel('Reflected Normal velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Normal Velocity Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()