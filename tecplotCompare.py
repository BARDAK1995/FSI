import pandas as pd
import numpy as np
casename = "case1_half"
casename = "jet4mm3"

file_path_main = f'./PM_BC_november/{casename}.tec'
# file_path_main31 = f'./Phonic_state_MS2/{casename}_half_modified.tec'
# file_path_main32 = f'./Phonic_state_MS2/{casename}_modified.tec'
file_path_ref = './PM_BC_november/refnp.tec'
new_file_path = f'./PM_BC_november/m2_{casename}.tec'

# file_path_ref = f'./Phonic_state_MS4/{casename}_half.tec'
# new_file_path = f'./Phonic_state_MS4/{casename}_modified_UNSTEADY.tec'

# file_path_ref = f'./Phonic_state_MS4/{casename}_otherhalf.tec'
# new_file_path = f'./Phonic_state_MS4/{casename}_modified_UNSTEADY2.tec'

# file_path_main = f'./Phonic_state_MS4/ref5.tec'
# file_path_ref = f'./Phonic_state_MS4/ref_half.tec'
# new_file_path = f'./Phonic_state_MS4/ref_modified_UNSTEADY.tec'

with open(file_path_main, 'r') as mainFile, open(file_path_ref, 'r') as refFile, open(new_file_path, 'w') as modified:
    # Writing the first and modified second lines to the new file
    first_line = mainFile.readline().strip()
    variables_line = mainFile.readline().strip()
    third_line = mainFile.readline().strip()
    refFile.readline()
    refFile.readline()
    refFile.readline()

    preview_lines = [first_line, variables_line, third_line]

    variables = variables_line.split('"')[1::2]  # Extracting the variable names
    press_column_index = variables.index("Press")
    Rho_column_index = variables.index("Rho  ")
    MA_column_index = variables.index("Ma   ") 
    T_column_index = variables.index("Ttrn ")
    V_column_index = variables.index("Vx   ") 

    # rhodif_index = variables.index("rho_dif") 
    # padif_index = variables.index("Pa_dif") 

    # Step 3: Preparing the modified second line to include "NEWVAR"
    modified_second_line = variables_line.strip() + ',      "rho_dif",      "Ma_dif,      "Pa_dif",      "T_dif",      "V_ref", ,      "Rho_ref",      "P_ref",      "waveSpeed"'
    # modified_second_line = variables_line.strip() + ',      "rho_dif",      "Ma_dif,      "Pa_dif",      "T_dif",      "V_ref", ,      "Rho_ref",      "P_ref",      "waveSpeed",      "rhoUNSTEADY",      "paUNSTEADY"'
    modified.write(first_line + '\n')
    modified.write(modified_second_line + '\n')
    modified.write(third_line + '\n')
    
    # Iterate through the remaining lines
    for line1, line2 in zip(mainFile, refFile):
        stripped_lineMain = line1.strip()
        stripped_lineRef = line2.strip()
        # If line starts with "ZONE", copy it as-is
        if stripped_lineMain.startswith("ZONE"):
            modified.write(stripped_lineMain + '\n')
        else:
            # Split the line into individual values
            mainValues = stripped_lineMain.split()
            refValues = stripped_lineRef.split()
            # ratio_pressure_percent = abs((float(mainValues[press_column_index]) - float(refValues[press_column_index])) / ((float(refValues[press_column_index]) + float(mainValues[press_column_index]) +1/100000000000)/2)*100)
            ratio_pressure_percent = (float(mainValues[press_column_index]) - float(refValues[press_column_index]))
            # ratio_pressure_percent = ((float(mainValues[press_column_index]) - float(refValues[press_column_index])) / ((float(refValues[press_column_index]) + float(mainValues[press_column_index]) +1/100000000000)/2)*100)
            ratio_Rho_percent = ((float(mainValues[Rho_column_index]) - float(refValues[Rho_column_index])) / ((float(refValues[Rho_column_index]) + float(mainValues[Rho_column_index]) +1/100000000000)/2)*100)
            ratio_Ma_percent = abs((float(mainValues[MA_column_index]) - float(refValues[MA_column_index])))
            ratio_T_percent = ((float(mainValues[T_column_index]) - float(refValues[T_column_index])) / ((float(refValues[T_column_index]) + float(mainValues[T_column_index]) +1/100000000000)/2)*100)
            
            #refs
            V_ref = float(refValues[V_column_index])
            rho_ref = float(refValues[Rho_column_index])
            P_ref = float(refValues[press_column_index])
            T_ref = float(refValues[T_column_index])
            a = np.sqrt(298.8 * T_ref)
            wavespeed = a + V_ref
            # Add the value for "NEWVAR" column and join them back
            # pa_unst = float(refValues[padif_index]) - float(refValues[padif_index])
            # rho_unst = float(refValues[rhodif_index]) - float(refValues[padif_index])

            rho_dif = 1
            ma_dif = 2
            modified_line = '\t'.join(mainValues + ["{:.6E}".format(ratio_Rho_percent)] + ["{:.6E}".format(ratio_Ma_percent)] + ["{:.6E}".format(ratio_pressure_percent)] + ["{:.6E}".format(ratio_T_percent)] + ["{:.6E}".format(V_ref)] + ["{:.6E}".format(rho_ref)] + ["{:.6E}".format(P_ref)] + ["{:.6E}".format(wavespeed)]) + '\n'
            modified.write(modified_line)

new_file_path