# registry_meds_summary.py

# %% ################# Import Libraries
# Standard Library Imports
from sys import path
from pathlib import Path
import pandas as pd

# Get the project root from the environment variable
project_root = Path(__file__).resolve().parents[3]
path.append(str(project_root))  # type: ignore

# Import setup_project_paths from setup_paths.py
from core.setup_paths import setup_project_paths
setup_project_paths()

from path_manager import PATH_DATAEXPORT
from data_manager import (dstype, dia, dem, idx, config, site_metadata, dia_summary)

#%% Load and process medication data
df = pd.read_parquet(project_root.parent / "Combo_Drugs.parquet")

#%% ######################## find targets
from utils.med_encoding_utils import create_patient_groups, label_drug_groups
def process_target_drugs(target_drugs, df, dem):
    # Create patient groups and label drug groups
    patient_groups = create_patient_groups(df, target_drugs)
    dem = label_drug_groups(dem, patient_groups)

    # Print the number of distinct patients for each drug
    for drug, patients in patient_groups.items():
        distinct_patients = set(patients)
        print(f"{drug}: {len(distinct_patients)} distinct patients")

    # Identify patients in multiple groups
    def find_multiple_groups(patient_groups):
        patient_to_groups = {}
        for group, patients in patient_groups.items():
            for patient in patients:
                if patient not in patient_to_groups:
                    patient_to_groups[patient] = []
                patient_to_groups[patient].append(group)

        multiple_groups = {patient: groups for patient, groups in patient_to_groups.items() if len(groups) > 1}
        return multiple_groups, patient_to_groups

    multiple_groups, patient_to_groups = find_multiple_groups(patient_groups)
    print(multiple_groups)
    return dem, patient_groups, multiple_groups

# Identify the drug combinations
from collections import Counter
def identify_combo_categories(multiple_groups):

    # Normalize the drug combinations (sort the drug names)
    normalized_combos = [' and '.join(sorted(drugs)) for drugs in multiple_groups.values()]

    # Count the occurrences of each combination
    combo_counts = Counter(normalized_combos)

    # Convert the counts to a DataFrame
    combo_counts_df = pd.DataFrame(combo_counts.items(), columns=['Combination', 'Number of Patients'])

    return combo_counts_df



# %% Second set of target drugs
target_drugs_2 = {
    "Prednisone": ["Prednisolone", "Rayos", "Sterapred", "Deltasone"],
    "Deflazacort": ["Emflaza", "Calcort", "Tencort", "Bencart"],
    "Vamorolone": ["Agamree", "VBP-15", "17α,21-Dihydroxy-16α-methylpregna-1,4,9(11)-triene-3,20-dione"],
}

dem, patient_groups, multiple_groups = process_target_drugs(target_drugs_2, df, dem)

combo_counts = identify_combo_categories(multiple_groups)

# %% First set of target drugs
target_drugs_1 = {
    "Edaravone": ["Edaravone", "Radicava"],
    "Riluzole": ["Riluzole", "Rilutek", "Tiglutik", "Exservan"],
    "Tofersen": ["Tofersen", "Qalsody"],
    "Nuedexta": ["Dextromethorphan", "Nuedexta"],
}

dem, patient_groups, multiple_groups = process_target_drugs(target_drugs_1, df, dem)

combo_counts = identify_combo_categories(multiple_groups)
combo_counts

#%% ################### Find Control Groups

drug_use_cols = list(target_drugs.keys())
drug_amenable_cols = [drug + ' Amenable' for drug in drug_use_cols]
total_patients = len(dem)


if dstype == 'DMD':
    from utils.exon_encoding_utils import label_amenable_groups, get_patient_list, add_amenable_columns_to_dem_table
    from plotly.subplots import make_subplots

    mut_cols = [
        'dnads', 'dnafam', 'dna', 'exontype', 'frametype', 'region', 'addmut', 
        'addmutoth', 'uexontype', 'uframetype', 'uexon', 'ufexon', 'utexon', 
        'uregion', 'ulistmut', 'fromexon', 'toexon'
    ]

    amenable_mut_types = ['Deletion', 'Non-sense mutation', 'Missense', 'Subexonic deletion']

    amenable_exons = {
        "Exondys 51 Amenable": [50, 52],
        "Vyondys 53 Amenable": [52, 54],
        "Viltepso 53 Amenable": [52, 54],
        "Amondys 45 Amenable": [44, 46],
        "AOC 1044 Amenable": [43, 45]
    }

    dia = label_amenable_groups(dia, amenable_exons, amenable_mut_types)
    amenable_patients = get_patient_list(dia, amenable_exons)
    dem = add_amenable_columns_to_dem_table(dem, amenable_patients)

if dstype == 'ALS':
    adults = idx[idx['Age'] >= 18]
    sod1_patients = dia[dia['genemut'].str.contains('SOD1', na=False)]
    num_sod1_patients = len(sod1_patients)
    print(f"Number of patients with SOD1 mutation: {num_sod1_patients}")

    adult_sod1_patients = sod1_patients[sod1_patients['FACPATID'].isin(adults['FACPATID'])]
    num_adult_sod1_patients = len(adult_sod1_patients)
    print(f"Number of adult patients with SOD1 mutation: {num_adult_sod1_patients}")

    # For Nuedexta, patients need to have 'Bulbar' in dia.bdypt.value_count
    dia = dia.drop(columns=['gender'], errors='ignore')
    dia = dia.merge(dem[['FACPATID', 'gender']], on='FACPATID', how='left')
    bulbar_patients = dia[dia['bdypt'].str.contains('Bulbar', na=False)]
    num_bulbar_patients = len(bulbar_patients)
    print(f"Number of patients with 'Bulbar' in bdypt: {num_bulbar_patients}")

    # Total number of bulbar patients by gender
    bulbar_patients_by_gender = bulbar_patients['gender'].value_counts()
    print("Number of bulbar patients by gender:")
    print(bulbar_patients_by_gender)

    adult_bulbar_patients = bulbar_patients[bulbar_patients['FACPATID'].isin(adults['FACPATID'])]
    num_adult_bulbar_patients = len(adult_bulbar_patients)
    print(f"Number of adult patients with 'Bulbar' in bdypt: {num_adult_bulbar_patients}")

    # Encode a column with the drug name + ' Amenable' if they can take it because their current age is 18
    for drug, aliases in target_drugs.items():
        amenable_col = f"{drug} Amenable"
        if drug == "Tofersen":
            # For Tofersen, it's just the ones that are over 18 and have the SOD1 mutation
            dem[amenable_col] = dem['FACPATID'].isin(adult_sod1_patients['FACPATID']).astype(int)
        elif drug == "Nuedexta":
            # For Nuedexta, it's the ones that are over 18 and have 'Bulbar' in bdypt
            dem[amenable_col] = dem['FACPATID'].isin(adult_bulbar_patients['FACPATID']).astype(int)
        else:
            # For other drugs, it's the ones that are over 18
            dem[amenable_col] = dem['FACPATID'].isin(adults['FACPATID']).astype(int)


# %% side histogram of Age
age_range = idx['Age'].agg(['min', 'max'])
print(f"Age range: {age_range['min']} - {age_range['max']}")

#%% Check for Specific Patient
# check = dia[dia['FACPATID'] == '9001-48'][mut_cols]
# check

#%% Create summary table
def create_summary_table(dem, drug_use_cols, drug_amenable_cols, total_patients):
    pivot_results = []
    
    for use_col, amenable_col in zip(drug_use_cols, drug_amenable_cols):
        if use_col not in dem.columns or amenable_col not in dem.columns:
            print(f"Column {use_col} or {amenable_col} not found in dem dataframe.")
            continue

        used_drug_count = (dem[use_col] == 1).sum()
        amenable_count = ((dem[amenable_col] == 1) & (dem[use_col] == 0)).sum()
        total_count = used_drug_count + amenable_count

        counts = {
            'Used Therapeutic Agent (N)': used_drug_count,
            'Not Using, But Eligible* (N)': amenable_count,
            'Total Eligible* (N)': total_count
        }

        crosstab = pd.Series(counts, name=use_col)
        pivot_results.append(crosstab)

    pivot_table = pd.concat(pivot_results, axis=1).T
    column_name = f'Eligible* Portion of MOVR Pop ({total_patients})'
    pivot_table[column_name] = (pivot_table['Total Eligible* (N)'] / total_patients) * 100
    pivot_table[column_name] = pivot_table[column_name].map('{:.2f}%'.format)
    
    return pivot_table



pivot_table = create_summary_table(dem, drug_use_cols, drug_amenable_cols, total_patients)
pivot_table
pivot_table.to_excel(PATH_DATAEXPORT / f'{dstype}_target_meds_summary.xlsx')
print(f"Summary table saved to {PATH_DATAEXPORT / f'{dstype}_target_meds_summary.xlsx'}")

# %%  Dataframe that can be used to create 

# join dem and dia
df_med = dem.copy()
# id also like to add State from site_metadata to df_med on the FACILITY_DISPLAY_ID
df_med = df_med.merge(site_metadata[['FACILITY_DISPLAY_ID', 'State']], on='FACILITY_DISPLAY_ID', how='left')

# %% Plotly State
import plotly.express as px

drug_pairs = list(zip(drug_use_cols, drug_amenable_cols))

from plotly.subplots import make_subplots
from collections import Counter
import plotly.graph_objects as go

fig = make_subplots(
    cols=len(drug_pairs), 
    rows=2, 
    subplot_titles=[f'{drug}' for drug, _ in drug_pairs] + [f'{amenable}' for _, amenable in drug_pairs],
    vertical_spacing=0.1,  # Adjust vertical spacing between plots
    horizontal_spacing=0.05  # Adjust horizontal spacing between plots
)

for i, (drug, amenable) in enumerate(drug_pairs, start=1):
    # Add histograms for drug and amenable columns
    fig.add_trace(
        go.Histogram(
            x=df_med[df_med[drug] == 1]['State'],
            name=f'{drug}',
            marker_color='blue',
            showlegend=False  # Remove legend
        ),
        row=1, col=i
    )
    # Add histograms for drug and amenable columns
    fig.add_trace(
        go.Histogram(
            # Filter for amenable but not using the drug
            x=df_med[df_med[amenable] == 1]['State'],
            name=f'{amenable}',
            marker_color='red',
            showlegend=False  # Remove legend
        ),
        row=2, col=i
    )

fig.update_layout(
    height=800,  # Adjust height to make the title higher
    width=1400,  # Adjust width to make the graphs wider
    title_text=f"<br>Distribution of {dstype} Participants by State: On Therapeutic vs. Eligible*",
    title_y=0.95,  # Adjust title position
    title_font=dict(size=16),  # Set title font size to 15
    font=dict(size=10)  # Adjust font size
)
fig.update_yaxes(range=[0, 500], title_text='Participants')
fig.show()

fig.write_image(PATH_DATAEXPORT / f'{dstype}_drug_distribution_by_state.png')
print(f"Figure saved to {PATH_DATAEXPORT / f'{dstype}_drug_distribution_by_state.png'}")


# %% END