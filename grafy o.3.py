import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

IV_data_file = ""
CA_data_file = ""
PEIS_data_file = ""
PEIS_data_files = []
IV_data_files_for_comparision = []
CA_data_files_list = []

   

def make_IVC_graph(IVC_data_file, data_dir):
    
    data_path = os.path.join(data_dir, IVC_data_file)
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']  # Add more encodings if needed
    for encoding in encodings:
        try:
            data = pd.read_csv(data_path, delimiter='\t', encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Unable to decode file using any of the specified encodings")

    # Ensure the required columns exist
    required_columns = ['<I>/mA', 'Ewe/V']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Required columns {missing_columns} not found in {IVC_data_file}")

    # Print the first 3 rows of the relevant data
    print(round(max(data['Ewe/V']),1),round(min(data['Ewe/V']),1))
    
    # Plot the I-V curve
    plt.figure(figsize=(10, 6))
    plt.plot(data['Ewe/V'], data['<I>/mA'], label=r'$I$($U$) [mA]')
    plt.xlabel(r'$U$ (V)')
    plt.ylabel(r'$I$ (mA)')
    #plt.title('I-V Curve')
    plt.legend()
    
    save_path = os.path.join(data_dir, f"IVC_graph_U_range {round(min(data['Ewe/V']),1)} - {round(max(data['Ewe/V']),1)}.png")
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")
    plt.show()
    
def make_CA_graph(CA_data_files, data_dirs):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']  # Add more encodings if needed
    labels = ['n-type', 'p-type']
    plt.figure(figsize=(10, 6))
    max_current = 0  # To find the maximum value of <I>/mA across all files
    
    for CA_data_file, data_dir, label in zip(CA_data_files, data_dirs, labels):
        data_path = os.path.join(data_dir, CA_data_file)
        for encoding in encodings:
            try:
                data = pd.read_csv(data_path, delimiter='\t', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError(f"Unable to decode file {CA_data_file} using any of the specified encodings")

        # Ensure the required columns exist
        required_columns = ['time/s', '<I>/mA']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Required columns {missing_columns} not found in {CA_data_file}")

        # Print the first 3 rows of the relevant data
        print(f"Data from {CA_data_file}:\n", data[required_columns].head(3))
        
        # Plot the I-t curve
        plt.plot(data['time/s'], data['<I>/mA'], label=f'{label} $I$ (t) [mA]')
        
        # Update the maximum current value
        max_current = max(max_current, data['<I>/mA'].max())

    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'$I$ (mA)')
    plt.ylim(0, max_current + 0.1)  # Set y-axis limits from 0 to max value + 3
    plt.legend()
    
    save_path = os.path.join(data_dirs[0], "CA_graph_comparison.png")  # Save in the first data directory
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")
    plt.show()
        
def make_Nyquist_graph(PEIS_data_files, data_dir):
    
    plt.figure(figsize=(10, 6))
    file_num = 0
    for file in PEIS_data_files:
        data_path = os.path.join(data_dir, file)
        
        # Try reading the file with different encodings
        encodings = ['latin1', 'ISO-8859-1', 'utf-16']
        for encoding in encodings:
            try:
                data = pd.read_csv(data_path, delimiter='\t', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("Unable to decode file using any of the specified encodings")
        
        # Filter columns containing real and imaginary parts of impedance Z
        impedance_columns = [col for col in data.columns if 'Re(Z)/Ohm' in col or '-Im(Z)/Ohm' in col]
        
        # Check if both real and imaginary parts are present
        if len(impedance_columns) != 2:
            raise ValueError("Expected columns for both real and imaginary parts of impedance Z not found")
        
        RE_impedance_column = [col for col in impedance_columns if 'Re(Z)/Ohm' in col][0]
        IM_impedance_column = [col for col in impedance_columns if '-Im(Z)/Ohm' in col][0]
        
        # Plot the Nyquist graph for each file
        plt.plot(data[RE_impedance_column], data[IM_impedance_column], marker='o', linestyle='', label=rf"Im($Z$)(Re($Z$)) at {file_num}V")
        file_num +=1
        
    plt.xlabel('Real(Z)/Ohm')
    plt.ylabel('-Imaginary(Z)/Ohm')
    plt.gca().invert_yaxis()
    #plt.title('Nyquist Plot')
    plt.legend(loc="upper right")
    plt.gca().invert_yaxis()  
    
    save_path = os.path.join(data_dir, "nyquist_graph.png")
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")
    plt.show()

def make_Bode_plot(PEIS_data_files, data_dir):
    plt.figure(figsize=(10, 6))
    
    # Initialize lists to store lines and labels for the legend
    lines = []
    labels = []
    
    # Plot the first file's data
    first_file = PEIS_data_files[0]
    first_data_path = os.path.join(data_dir, first_file)
    for encoding in ['latin1', 'ISO-8859-1', 'utf-16']:
        try:
            first_data = pd.read_csv(first_data_path, delimiter='\t', encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Unable to decode file using any of the specified encodings")
    
    first_data['log(freq/Hz)'] = np.log10(first_data['freq/Hz'])
    
    # Print the first three rows of the relevant data for plotting
    #print(f"First three rows of plotted data from {first_file}:\n{first_data[['log(freq/Hz)', '|Z|/Ohm', 'Phase(Z)/deg']].head(3)}\n")
    
    # Plot magnitude vs. log(frequency) on the left y-axis
    ax1 = plt.gca()
    line1, = ax1.plot(first_data['log(freq/Hz)'], first_data['|Z|/Ohm'], label='Impedance at 0V', color='blue')
    ax1.set_xlabel(r'log($f$)')
    ax1.set_ylabel(r'|$Z$| [$\Omega$]', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a twin Axes sharing the x-axis for phase on the right y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(first_data['log(freq/Hz)'], first_data['Phase(Z)/deg'], label='Phase at 0V', color='blue', linestyle='--')
    ax2.set_ylabel(r'Phase($Z$) [$^{\circ}$]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add the lines and labels to the lists
    lines.extend([line1, line2])
    labels.extend([line.get_label() for line in [line1, line2]])
    
    # Plot data from subsequent files on the same axes
    for file in PEIS_data_files[1:]:
        data_path = os.path.join(data_dir, file)
        for encoding in ['latin1', 'ISO-8859-1', 'utf-16']:
            try:
                data = pd.read_csv(data_path, delimiter='\t', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("Unable to decode file using any of the specified encodings")
        
        # Calculate log(frequency) for each file
        data['log(freq/Hz)'] = np.log10(data['freq/Hz'])
        
        # Print the first three rows of the relevant data for plotting
        #print(f"First three rows of plotted data from {file}:\n{data[['log(freq/Hz)', '|Z|/Ohm', 'Phase(Z)/deg']].head(3)}\n")
        
        # Plot magnitude vs. log(frequency) on the left y-axis
        line3, = ax1.plot(data['log(freq/Hz)'], data['|Z|/Ohm'], label='Impedance at 1V', color='red')
        
        # Plot phase vs. log(frequency) on the right y-axis
        line4, = ax2.plot(data['log(freq/Hz)'], data['Phase(Z)/deg'], label='Phase at 1V', color='red', linestyle='--')
        
        # Add the lines and labels to the lists
        lines.extend([line3, line4])
        labels.extend([line.get_label() for line in [line3, line4]])
    
    plt.tight_layout()
    ax1.legend(lines, labels, loc="center left")
    save_path = os.path.join(data_dir, "bode_graph.png")
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")
    plt.show()


def porovnej_IVC(file_paths,labels):
    def read_iv_data(file_path):
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']
        for encoding in encodings:
            try:
                data = pd.read_csv(file_path, delimiter='\t', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("Unable to decode file using any of the specified encodings")
        
        # Ensure the required columns exist
        required_columns = ['<I>/mA', 'Ewe/V']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if (missing_columns):
            raise ValueError(f"Required columns {missing_columns} not found in {file_path}")
        
        return data['<I>/mA'], data['Ewe/V']

    plt.figure(figsize=(10, 6))
    koef = 1
    počítání = 0
    for i, file_path in enumerate(file_paths):
        počítání +=1 
        print(počítání)
        
        currents, voltages = read_iv_data(file_path)
        if počítání ==2:
            currents = currents*koef
        plt.plot(voltages, currents, label=f'{labels[i]}')
    
    plt.xlabel(r'$U$ [V]')
    plt.ylabel(r'$I$ [mA]')
    #plt.yscale('log')
    plt.legend()
    
    output_dir = "G:\\Můj disk\\stáž\\data_k_porovnání_IVC"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ivc_comparison_for_ZnO_samples_30.png")
   
    
    plt.savefig(output_path)
    plt.show()





data_dirs = [
    r'G:\Můj disk\stáž\August 30 2024\ZnO n-type\data txt',
    r'G:\Můj disk\stáž\August 30 2024\ZnO p-type\data txt',
]


SLOŽKY_K_POROVNÁNÍ = ['ZnO n-type_05_CV_C01', 'ZnO p-type ya svetla_05_CV_C01']
file_paths = [rf'{data_dirs[0]}\{SLOŽKY_K_POROVNÁNÍ[0]}.txt', rf'{data_dirs[1]}\{SLOŽKY_K_POROVNÁNÍ[1]}.txt',]
labels = [r'$\mathrm{ZnO}$/Si n-type', r'$\mathrm{ZnO}$/Si p-type', r'$\mathrm{Cu_3N}$/glass']
porovnej_IVC(file_paths, labels)

počet_grafů = 0
počítání_ac = 0
for data_dir in data_dirs:
    files = os.listdir(data_dir)
    
    počítání = 0
    
    PEIS_data_files = []

    
    
    for file in files:
        if file.endswith('.txt'):
            if 'PEIS' in file:
                PEIS_data_file = file
                #print(PEIS_data_file)
                PEIS_data_files.append(file)
                počítání += 1
                if počítání == 2:
                    make_Nyquist_graph(PEIS_data_files, data_dir)
                    make_Bode_plot(PEIS_data_files, data_dir)
                    a=1
                    
                počet_grafů +=1
                #print(počet_grafů)
                continue  

            if 'CA' in file:
                CA_data_file = file
                print(CA_data_file)
                CA_data_files_list.append(CA_data_file)
                počítání_ac +=1
                if počítání_ac == 3:
                    make_CA_graph(CA_data_files_list, data_dirs)
                #počet_grafů +=1
                #print(počet_grafů)
                continue  
            
            if 'IVC' or 'IV char' in file:
                IV_data_files_for_comparision.append(file)

            if 'CV' or 'IV' in file:
                IV_data_file = file
                #print(IV_data_file)
                make_IVC_graph(IV_data_file, data_dir)
                počet_grafů +=1
                #print(počet_grafů)
                continue 
            
     


