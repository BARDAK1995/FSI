import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Computer Modern Roman"],
})
amuKg = 1.66e-27
dict_elements = {}
dict_elements["he"] = 4.0026 * amuKg
dict_elements["N2"] = 2*14 * amuKg

def maxwellian_densityWallSPEED_EPIC_first(C, m, T,A , k=1.38064852e-23):
    B = m / (k*T)
    constant_factor =  B * C
    UglyTerm = (1 - (B*A*C + 1)*np.exp(-B*A*C))/(B*A*A) + (1-np.exp(-B*A*C))
    # constant_factor = ((m / (2 * np.pi * k * T)) ** 1.5) * 4 * np.pi * v * v
    exponential_factor = np.exp(-B/2 * (C**2 + A**2))
    mean_velocity =0
    mostProbSpeed =0
    # print(f"Mean speed is {mean_velocity}")
    # print(f"most prob speed is {mostProbSpeed}")
    return constant_factor * exponential_factor * UglyTerm, mean_velocity

def maxwellian_densityWallSPEED_EPIC(C, m, T,A , k=1.38064852e-23):
    B = m / (k*T)
    constant_factor =  B*B * C*C
    UglyTerm = (np.exp(A*B*C)*(-A*A*B + A*B*C - 1) + np.exp(A*A*B))/(A*A*B*B*C)
    # constant_factor = ((m / (2 * np.pi * k * T)) ** 1.5) * 4 * np.pi * v * v
    exponential_factor = np.exp(-B/2 * (C**2 + A**2))
    mean_velocity = 0
    mostProbSpeed = 0
    # print(f"Mean speed is {mean_velocity}")
    # print(f"most prob speed is {mostProbSpeed}")
    return constant_factor * exponential_factor * UglyTerm, mean_velocity

def maxwellian_densityWall(v, m, T, k=1.38064852e-23):
    # constant_factor = ((m / (2 * np.pi * k * T)) ** 1.5) * 4 * np.pi * v * v
    constant_factor = (1/2) * (m / ( k * T))**2  * (v**3) 
    # constant_factor = ((m / (2 * np.pi * k * T)) ** 1.5) * 4 * np.pi * v * v

    # constant_factor = (1) * (m / ( k * T))  * (v) 
    exponential_factor = np.exp(-m * (v**2) / (2 * k * T))
    mean_velocity = np.sqrt((9/8) * np.pi * k * T / m )
    mostProbSpeed = 1.41 * (1.38064852e-23 * T / m) ** 0.5
    print(f"Mean speed is {mean_velocity}")
    print(f"most prob speed is {mostProbSpeed}")
    return constant_factor * exponential_factor, mean_velocity


def maxwellian_density1D(v, m, T, k=1.38064852e-23):
    # constant_factor = ((m / (2 * np.pi * k * T)) ** 1.5) * 4 * np.pi * v * v
    constant_factor =  (m / (2*np.pi  *  k * T))**0.5 

    exponential_factor = np.exp(-m *(v**2) / (2 * k * T))
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
unsteadyV = -150.5
A = unsteadyV
v_values = np.linspace(0, 1400, 1000)  # Generates 1000 points between 0 and 3000
v_valuesFull = np.linspace(-1400, 1400, 1000)  # Generates 1000 points between 0 and 3000

Speed_values, mean_VelocityS = maxwellian_densityWall(v_values, m, T)

Speed_valuesMinus, mean_VelocityS = maxwellian_densityWallSPEED_EPIC(v_values, m, T,A)
Speed_valuesPlus, mean_VelocityS = maxwellian_densityWallSPEED_EPIC(v_values, m, T,-A)


V_normalValues, mean_Velocity = maxwellian_probabilityFunc_NORMALVEL(v_values, m, T)

V_tangentValues, mean_Velocityt = maxwellian_density1D(v_valuesFull, m, T)

plt.figure(figsize=(12, 6))

plt.plot(v_values, V_normalValues, label=f'Nitrogen Gas, T = {T}K', linewidth=3, color='black')  # Increased linewidth
# plt.plot(v_values - unsteadyV, V_normalValues, label=f'v - unsteadyV, m={m}, T={T}', linewidth=2)
# plt.plot(v_values + unsteadyV, V_normalValues, label=f'v + unsteadyV, m={m}, T={T}', linewidth=2)

plt.xlabel('Reflected Normal velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Normal Velocity Probability Distribution', fontsize=20)  # Increased fontsize

# Adding an annotation inside the plot
plt.annotate(f'Most Probable Normal Velocity = {mean_Velocity:.1f} m/s', 
             xy=(max(v_values), max(V_normalValues)), 
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
plt.plot(v_values, V_normalValues, label='$f(V_n)$', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
plt.plot(v_values - unsteadyV, V_normalValues, label='$f(V_{n}^{PM})^{-} \mapsto V_{n}^{PM} = V_n - (A \omega)$', linewidth=3, color='red')
plt.plot(v_values + unsteadyV, V_normalValues, label='$f(V_{n}^{PM})^{+} \mapsto  V_{n}^{PM} = V_n + (A \omega)$', linewidth=3, color='blue')

plt.xlabel('Reflected Normal Velocity $V_n$ (m/s)', fontsize=20)  # Increased fontsize
plt.ylabel('Probability $f(V_n)$', fontsize=20)  # Increased fontsize
plt.title('Normal Velocity Probability Distribution', fontsize=20)  # Increased fontsize

# Adding an annotation inside the plot
# plt.annotate(f'Most Probable Normal Velocity = {mean_Velocity:.1f}±{unsteadyV}m/s', 
#              xy=(max(v_values), max(V_normalValues)), 
#              xytext=(-8, -60), 
#              textcoords='offset points',
#              horizontalalignment='right',
#              verticalalignment='top',
#              fontsize=16, 
#              color='Black',
#              bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))  # Box added here


plt.legend(fontsize=20)  # Increased fontsize
plt.grid(True)  
plt.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # y-axis in scientific notation
plt.xlim(0, 1400)
plt.ylim(0,0.0022)
plt.show()
# plt.tight_layout()
plt.show()



file_path1 = 'PM_BC2/ANAN1_0.fdf'
file_path2 = 'PM_BC2/ANAN2_0.fdf'
df1 = pd.read_csv(file_path1, delim_whitespace=True, header=None)
df2 = pd.read_csv(file_path2, delim_whitespace=True, header=None)
# Extract the 3rd column (index 2)
Vnormal_1 = df1[2] 
Vnormal_2 = df2[2]
Vtangent_1 = df1[3]
Vtangent_2 = df2[3]


Speed_1 = np.sqrt(Vnormal_1**2 +  Vtangent_1**2+Vtangent_2**2)
Speed_2 = np.sqrt(Vnormal_2**2 +  Vtangent_2**2 + Vtangent_1**2)

# Create a histogram
plt.figure(figsize=(10, 6))
# plt.hist(Vnormal_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
plt.hist(Vnormal_2, bins=50,density=True,color='red', edgecolor='black',alpha=0.5, label='Vmin')
plt.plot(v_values, V_normalValues, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
plt.plot(v_values - unsteadyV, V_normalValues, label='Vmin', linewidth=3, color='red')
# plt.plot(v_values + unsteadyV, V_normalValues, label='Vmax', linewidth=3, color='blue')
plt.xlabel('Reflected Normal velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Normal Velocity Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(Vnormal_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
# plt.hist(Vnormal_2, bins=50,density=True,color='red', edgecolor='black',alpha=0.5, label='Vmin')
plt.plot(v_values, V_normalValues, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
# plt.plot(v_values - unsteadyV, V_normalValues, label='Vmin', linewidth=3, color='red')
plt.plot(v_values + unsteadyV, V_normalValues, label='Vmax', linewidth=3, color='blue')
plt.xlabel('Reflected Normal velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Normal Velocity Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(Vnormal_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
plt.hist(Vnormal_2, bins=50,density=True,color='red', edgecolor='black',alpha=0.5, label='Vmin')
plt.plot(v_values, V_normalValues, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
plt.plot(v_values - unsteadyV, V_normalValues, label='Vmin', linewidth=3, color='red')
plt.plot(v_values + unsteadyV, V_normalValues, label='Vmax', linewidth=3, color='blue')
plt.xlabel('Reflected Normal velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Normal Velocity Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()




# Create a histogram
plt.figure(figsize=(10, 6))
# plt.hist(Vnormal_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
plt.hist(Vtangent_2, bins=50,density=True,color='red', edgecolor='black',alpha=0.5, label='Vmin')
plt.hist(Vtangent_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')

plt.plot(v_valuesFull, V_tangentValues, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
# plt.plot(v_values + unsteadyV, V_normalValues, label='Vmax', linewidth=3, color='blue')
plt.xlabel('Reflected Tangential velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Tangential Velocity Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()

# plt.figure(figsize=(10, 6))
# plt.hist(Vtangent_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
# # plt.hist(Vnormal_2, bins=50,density=True,color='red', edgecolor='black',alpha=0.5, label='Vmin')
# plt.plot(v_values, V_normalValues, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
# # plt.plot(v_values - unsteadyV, V_normalValues, label='Vmin', linewidth=3, color='red')
# plt.plot(v_values + unsteadyV, V_normalValues, label='Vmax', linewidth=3, color='blue')
# plt.xlabel('Reflected Normal velocity(m/s)', fontsize=18)  # Increased fontsize
# plt.ylabel('Probability', fontsize=18)  # Increased fontsize
# plt.title('Normal Velocity Probability Distribution', fontsize=20)  # Increased fontsize
# plt.legend(fontsize=14)  # Increased fontsize
# plt.show()

# Create a histogram
plt.figure(figsize=(10, 6))
# plt.hist(Vnormal_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
plt.hist(Speed_2, bins=50,density=True,color='red', edgecolor='black',alpha=0.5, label='Vmin')
plt.hist(Speed_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')

plt.plot(v_values, Speed_values, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth

plt.xlabel('Reflected SPEED(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Reflected particle SPEED Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()


plt.figure(figsize=(10, 6))
# plt.hist(Vnormal_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
plt.hist(Speed_2, bins=50,density=True,color='red', edgecolor='black',alpha=0.5, label='Vmin')
plt.plot(v_values, Speed_values, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth

plt.xlabel('Reflected SPEED(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Reflected particle SPEED Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()

plt.figure(figsize=(10, 6))
# plt.hist(Vnormal_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
plt.hist(Speed_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')

plt.plot(v_values, Speed_values, label='steady', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth

plt.xlabel('Reflected SPEED(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Reflected particle SPEED Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()

plt.figure(figsize=(10, 6))
# plt.hist(Vnormal_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
plt.plot(v_values, Speed_values, label='Reflected Particle SPEED', linewidth=3, color='black')  # Increased linewidth
plt.plot(v_valuesFull, V_tangentValues, label='Tangential Velocity', linewidth=1, color='red')  # Increased linewidth
plt.plot(v_values, V_normalValues, label='Normal Velocity', linewidth=1, color='blue')  # Increased linewidth

plt.xlabel('velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Probability Distribution Functions', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()


plt.figure(figsize=(10, 6))
# plt.hist(Vnormal_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')
plt.plot(v_values, Speed_values, label='Reflected Particle SPEED', linewidth=1, color='black')  # Increased linewidth
plt.plot(v_values, Speed_valuesPlus, label='Tangential Velocity', linewidth=1, color='red')  # Increased linewidth
plt.plot(v_values, Speed_valuesMinus, label='Normal Velocity', linewidth=1, color='blue')  # Increased linewidth

plt.xlabel('velocity(m/s)', fontsize=18)  # Increased fontsize
plt.ylabel('Probability', fontsize=18)  # Increased fontsize
plt.title('Probability Distribution Functions', fontsize=20)  # Increased fontsize
plt.legend(fontsize=14)  # Increased fontsize
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(v_values, Speed_values, label='$f(C)$', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
plt.plot(v_values, Speed_valuesMinus, label='$f(C^{PM})^{-} \mapsto C^{PM} = C - A$', linewidth=3, color='red')
plt.plot(v_values, Speed_valuesPlus, label='$f(C^{PM})^{+} \mapsto  C^{PM} = C + A$', linewidth=3, color='blue')

plt.xlabel('Reflected Speed $C$ (m/s)', fontsize=20)  # Increased fontsize
plt.ylabel('Probability $f(C)$', fontsize=20)  # Increased fontsize
plt.title('PM Modified Speed Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=20)  # Increased fontsize
plt.grid(True)  
plt.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # y-axis in scientific notation
plt.xlim(0, 1400)
plt.ylim(0,0.0022)
plt.show()
# plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 6))
plt.plot(v_values, Speed_values, label='$f(C)$', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
plt.plot(v_values, Speed_valuesMinus, label='$f(C^{PM})^{-} \mapsto C^{PM} = C - A$', linewidth=3, color='red')
plt.plot(v_values, Speed_valuesPlus, label='$f(C^{PM})^{+} \mapsto  C^{PM} = C + A$', linewidth=3, color='blue')
plt.hist(Speed_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')

plt.xlabel('Reflected Speed $C$ (m/s)', fontsize=20)  # Increased fontsize
plt.ylabel('Probability $f(C)$', fontsize=20)  # Increased fontsize
plt.title('PM Modified Speed Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=20)  # Increased fontsize
plt.grid(True)  
plt.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # y-axis in scientific notation
plt.xlim(0, 1400)
plt.ylim(0,0.0022)
plt.show()
# plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 6))
plt.plot(v_values, Speed_values, label='$f(C)$', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
#plt.plot(v_values, Speed_valuesMinus, label='$f(C^{PM})^{-} \mapsto C^{PM} = C - A$', linewidth=3, color='red')
plt.plot(v_values, Speed_valuesPlus, label='$f(C^{PM})^{+} \mapsto  C^{PM} = C + A$', linewidth=3, color='blue')
plt.hist(Speed_1, bins=50,density=True,color='blue', edgecolor='black',alpha=0.5, label='Vmax')

plt.xlabel('Reflected Speed $C$ (m/s)', fontsize=20)  # Increased fontsize
plt.ylabel('Probability $f(C)$', fontsize=20)  # Increased fontsize
plt.title('PM Modified Speed Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=20)  # Increased fontsize
plt.grid(True)  
plt.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # y-axis in scientific notation
plt.xlim(0, 1400)
plt.ylim(0,0.0022)
plt.show()
# plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(v_values, Speed_values, label='$f(C)$', linewidth=3, color='black', linestyle='dotted')  # Increased linewidth
plt.plot(v_values, Speed_valuesMinus, label='$f(C^{PM})^{-} \mapsto C^{PM} = C - A$', linewidth=3, color='red')
#plt.plot(v_values, Speed_valuesPlus, label='$f(C^{PM})^{+} \mapsto  C^{PM} = C + A$', linewidth=3, color='blue')
plt.hist(Speed_2, bins=50,density=True,color='red', edgecolor='black',alpha=0.5, label='Vmin')

plt.xlabel('Reflected Speed $C$ (m/s)', fontsize=20)  # Increased fontsize
plt.ylabel('Probability $f(C)$', fontsize=20)  # Increased fontsize
plt.title('PM Modified Speed Probability Distribution', fontsize=20)  # Increased fontsize
plt.legend(fontsize=20)  # Increased fontsize
plt.grid(True)  
plt.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # y-axis in scientific notation
#plt.xlim(0, 1400)
#plt.ylim(0,0.0022)
plt.show()
# plt.tight_layout()
plt.show()
