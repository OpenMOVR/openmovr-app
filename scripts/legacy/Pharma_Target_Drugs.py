# %% Set up 
# Import necessary modules
from sys import path
from pathlib import Path
import pandas as pd

# Get the project root from the environment variable
project_root = Path(__file__).resolve().parents[3]
path.append(str(project_root))  # type: ignore

# Import setup_project_paths from setup_paths.py
from core.setup_paths import setup_project_paths

# Set up project paths
setup_project_paths()

from path_manager import PATH_DATA_AGG, PATH_SHARABLES
from data_manager import (config, list_delements,
                          data_dict, #filtered Data Dictionary  as pd.Dataframe
                          movr_dict, #data dictionary excel doc as a pd.Dataframe
                          database, #all data
                          data, #dict of pd.Dataframes that have  initial config details
                          data_to_filter, # data + additional filtering for (age,weight) and 
                          df, df_idx, df_filtered,  #merged and filtered pd.Dataframe from config
                          dc,)  #data columns of interest - to be edited.
                           
from data_processing import (
                            clean_data_load_main,   # applied config settings (dstype[''])
                            data_loader_main        # doesn't applied config settings; defaults to no filters (filter_dstype=[''])
                            )

from utils import find_and_print_elements as fpe
from utils.data_dictionary_utils import create_data_dictionary
from utils.export_utils import export_clean_data_to_excel
from epidemiology import epidemiology_utils as epi_utils

hub, dia, enc, dem, med, idx = fpe.tuple_of_main_dataframes(database)
#%%
print("All Data loaded successfully and ready for processing as 'database' \n")
print(data.get('summary'))
#%%
len(med)
#%% clean up meds
#%% run clean up on Med columns
from utils.rx_standardization_utils import combine_medications, explode_medications, apply_standard_names, drop_duplicates, process_medication_data

df = process_medication_data(med)
df

#%%
# Create a dictionary to store FACPATID lists for each drug group
from utils.med_encoding_utils import create_patient_groups, label_drug_groups

# Ensure target_drugs is a dictionary to feed into functions
targets = {
    "ALS": ["riluzole", "Rilutek", "edaravone", "Radicava", "tofersen", "Qalsody"],
    "SMA": ["risdiplam", "Evrysdi", "nusinersen", "Spinraza", "onasemnogene abeparvovec", "Zolgensma"]
}

# Iterate over dictionary items
for disease, drug_list in targets.items():
    patients = dem[dem['dstype'] == disease]['FACPATID'].unique()  # Get unique patient IDs
    disease_meds = df[df['FACPATID'].isin(patients)]
    disease_pop = len(disease_meds['FACPATID'].unique())

    # Create patient groups based on target drugs
    target_drugs_dict = {disease: {disease: drug_list}}  # Wrap the drug_list inside a dictionary
    patient_groups = create_patient_groups(disease_meds, target_drugs_dict)

    # Label demographic dataframe with drug groups
    dem = label_drug_groups(dem, patient_groups)

    # Store results
    results[disease] = {
        'patient_list': patients,
        'medications': disease_meds,
        'Disease N': disease_pop,
        'patient_groups': patient_groups,
        'dem': dem.copy()  # Store a copy to prevent overwriting
    }

    print(f"{disease} Patients with Medications Listed: {disease_pop}")

# Debugging output
print("Results keys:", results.keys())  # Should show 'ALS' and 'SMA'


#%% display number of participants in each group
for drug, patients in patient_groups.items():
    # Count distinct patient IDs using a set
    distinct_patients = set(patients)
    print(f"{drug}: {len(distinct_patients)} distinct patients")

#%% ######################## find targets