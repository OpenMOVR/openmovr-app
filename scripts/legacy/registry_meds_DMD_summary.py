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
from data_manager import (dstype, dia, dem, config, site_metadata)

#%% Load and process medication data
df = pd.read_parquet(project_root.parent / "Combo_Drugs.parquet")

#%% ######################## find targets
from utils.med_encoding_utils import create_patient_groups, label_drug_groups

target_drugs = {
    "Exondys 51": ["Exondys", "Eteplirsen"],
    "Vyondys 53": ["Golodirsen", "Vyondys"],
    "Viltepso 53": ["Viltepso", "Viltolarsen"],
    "Amondys 45": ["Casimersen", "Amondys"],
    "AOC 1044": ["AOC", "44 Skipping"]
}

target_drugs = {
    "Prednisone": ["Prednisolone", "Rayos", "Sterapred", "Deltasone"],
    "Deflazacort": ["Emflaza", "Calcort", "Tencort", "Bencart"],
    "Vamorolone": ["Agamree", "VBP-15", "17α,21-Dihydroxy-16α-methylpregna-1,4,9(11)-triene-3,20-dione"],
}

patient_groups = create_patient_groups(df, target_drugs)
dem = label_drug_groups(dem, patient_groups)

for drug, patients in patient_groups.items():
    distinct_patients = set(patients)
    print(f"{drug}: {len(distinct_patients)} distinct patients")

# %% Identify patients in multiple groups
def find_multiple_groups(patient_groups):
    patient_to_groups = {}
    for group, patients in patient_groups.items():
        for patient in patients:
            if patient not in patient_to_groups:
                patient_to_groups[patient] = []
            patient_to_groups[patient].append(group)
    
    multiple_groups = {patient: groups for patient, groups in patient_to_groups.items() if len(groups) > 1}
    return multiple_groups, patient_to_groups

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

multiple_groups, patient_to_groups = find_multiple_groups(patient_groups)
multiple_groups

combo_counts = identify_combo_categories(multiple_groups)
combo_counts

#%% ################### Find Control Groups
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

#%% Check for Specific Patient
check = dia[dia['FACPATID'] == '9001-48'][mut_cols]
check

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
            'Used Skipping Agent (N)': used_drug_count,
            'Not Using, But Amenable (N)': amenable_count,
            'Total Amenable (N)': total_count
        }

        crosstab = pd.Series(counts, name=use_col)
        pivot_results.append(crosstab)

    pivot_table = pd.concat(pivot_results, axis=1).T
    column_name = f'Amenable Portion of MOVR Pop ({total_patients})'
    pivot_table[column_name] = (pivot_table['Total Amenable (N)'] / total_patients) * 100
    pivot_table[column_name] = pivot_table[column_name].map('{:.2f}%'.format)
    
    return pivot_table

drug_use_cols = ['Exondys 51', 'Vyondys 53', 'Viltepso 53', 'Amondys 45', 'AOC 1044']
drug_amenable_cols = [
    'Exondys 51 Amenable', 'Vyondys 53 Amenable', 'Viltepso 53 Amenable', 
    'Amondys 45 Amenable', 'AOC 1044 Amenable'
]
total_patients = len(dem)

pivot_table = create_summary_table(dem, drug_use_cols, drug_amenable_cols, total_patients)
pivot_table
pivot_table.to_excel(PATH_DATAEXPORT / f'{dstype}_target_meds_summary.xlsx')
print(f"Summary table saved to {PATH_DATAEXPORT / f'{dstype}_target_meds_summary.xlsx'}")

# %%  Dataframe that can be used to create 

# join dem and dia
df_med = dem.join(dia[['dna', 'exontype', 'frametype', 'fromexon', 'toexon']])
# id also like to add State from site_metadata to df_med on the FACILITY_DISPLAY_ID
df_med = df_med.merge(site_metadata[['FACILITY_DISPLAY_ID', 'State']], on='FACILITY_DISPLAY_ID', how='left')

# %% Plotly State
import plotly.express as px

drug_pairs = [
    ('Exondys 51', 'Exondys 51 Amenable'),
    ('Vyondys 53', 'Vyondys 53 Amenable'),
    ('Viltepso 53', 'Viltepso 53 Amenable'),
    ('Amondys 45', 'Amondys 45 Amenable'),
    ('AOC 1044', 'AOC 1044 Amenable')
]

from plotly.subplots import make_subplots
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
    title_text=f"<br>Distribution of {dstype} Participants by State: On Therapeutic vs. Amenable",
    title_y=0.95,  # Adjust title position
    title_font=dict(size=16),  # Set title font size to 15
    font=dict(size=10)  # Adjust font size
)
fig.update_yaxes(range=[0, 35], title_text='Participants')
fig.show()

fig.write_image(PATH_DATAEXPORT / 'DMD_drug_distribution_by_state.png')
print(f"Figure saved to {PATH_DATAEXPORT / 'DMD_drug_distribution_by_state.png'}")


# %% END