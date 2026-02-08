# registry_meds_summary.py
# check 3/24/2025 - adp
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
from data_manager import (dstype, dia, dem, idx,enc,
                          config, site_metadata, 
                          dia_summary, enc_summary)

#%% Load and process medication data
# load a standardized table of participant medication records
df = pd.read_parquet(project_root.parent / "Combo_Drugs.parquet")

# encounter by last last by encntdt 
enc_last = enc.sort_values(by='encntdt', ascending=False).drop_duplicates(subset='FACPATID', keep='first')


#%% ######################## Find targets
# functions to process target drugs and idnetify combo categories
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

# %% Target drugs
# target drug labeling
target_drug = 'steriod'   
target_drugs = config.get('drugs')[dstype]
steriod = config.get('drugs')['Steriod']

if target_drug == 'steriod':
    target_drugs = steriod
elif target_drug == 'combo':
    target_drugs.update(steriod)

dem, patient_groups, multiple_groups = process_target_drugs(target_drugs, df, dem)

# Print the key and the total count per patient group


combo_counts = identify_combo_categories(multiple_groups)
combo_counts = combo_counts.sort_values('Number of Patients', ascending=False)

# Print the total count of patients for each drug
for drug, patients in patient_groups.items():
    if drug not in ['Exondys 51', 'Vyondys 53', 'Viltepso 53', 'Amondys 45', 'AOC 1044']:
        total_count = len(set(patients))
        print(f"{drug}: {total_count} total patients")

# Add a row for patients not in any combo group
# not_combo_patients_count = len(dem) - len(multiple_groups)
# combo_counts = combo_counts.append({'Combination': 'Not a combo patient in MOVR', 'Number of Patients': not_combo_patients_count}, ignore_index=True)

display(combo_counts.style.hide(axis='index'))
# combo_counts.sort_values('Number of Patients', ascending=False).to_excel(PATH_DATAEXPORT / f'{dstype}_combo_counts.xlsx')

#%% ################### Find Control Groups

# find control groups and append dem to drug and eligible encodings for participants
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
        elif drug in ['Edaravone','Riluzole']:
            # For other drugs, it's the ones that are over 18
            dem[amenable_col] = dem['FACPATID'].isin(adults['FACPATID']).astype(int)
        else:
            # For other drugs it's for all ages
            dem[amenable_col] = 1

if dstype == 'SMA':
    # Calculate adults
    adults = idx[idx['Age'] >= 18]

    # Encode a column with the drug name + ' Amenable' if they can take it because their current age is 18
    for drug, aliases in target_drugs.items():
        amenable_col = f"{drug} Amenable"
        if drug == "Spinraza":
            # For Spinraza, it's the ones that are over 18
            dem[amenable_col] = dem['FACPATID'].isin(adults['FACPATID']).astype(int)
        elif drug == "Zolgensma":
            # For Zolgensma, it's the ones that are under 2
            dem[amenable_col] = dem['FACPATID'].isin(idx[idx['Age'] < 2]['FACPATID']).astype(int)
        elif drug == "Evrysdi":
            # For Evrysdi, it's the ones that are 2 months and older
            dem[amenable_col] = dem['FACPATID'].isin(idx[idx['Age'] >= (2/12)]['FACPATID']).astype(int)
        else:
            # For other drugs, it's for all ages
            dem[amenable_col] = 1


# %% Check Histogram of Age
# calculate and plot age range
age_range = idx['Age'].agg(['min', 'max'])
print(f"Age range: {age_range['min']} - {age_range['max']}")

#%% Create Summary Med Table as pivot_table pd.DataFrame()

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
    column_name = f'MOVR Participants with Therapy Use (%)'
    pivot_table[column_name] = (pivot_table['Used Therapeutic Agent (N)'] / total_patients) * 100
    pivot_table[column_name] = pivot_table[column_name].map('{:.2f}%'.format)
    return pivot_table

pivot_table = create_summary_table(dem, drug_use_cols, drug_amenable_cols, total_patients)
display(pivot_table)
# pivot_table.to_excel(PATH_DATAEXPORT / f'{dstype}_target_meds_summary.xlsx')
# print(f"Summary table saved to {PATH_DATAEXPORT / f'{dstype}_target_meds_summary.xlsx'}")

# %% PIVOT TABLE Graph

# I need percentage of patients using the drug and percentage of patient amenable to the drug in a graph

import matplotlib.pyplot as plt

def plot_percentage_distribution(pivot_table):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the percentage of patients using the drug
    pivot_table['MOVR Participants with Therapy Use (%)'].str.rstrip('%').astype(float).plot(kind='bar', ax=ax, color='blue', alpha=0.7, position=1, width=0.4)

    # Plot the percentage of patients amenable to the drug
    pivot_table['Eligible* Portion of MOVR Pop ({})'.format(total_patients)].str.rstrip('%').astype(float).plot(kind='bar', ax=ax, color='red', alpha=0.7, position=0, width=0.4)

    ax.set_title('Percentage of DMT Patients:  Using vs Eligible', fontsize=24)
    ax.set_ylabel('Percentage (%)', fontsize=21)
    # ax.set_xlabel('Drugs', fontsize=14)
    ax.legend(['Using DMT', 'Eligible'], fontsize=21)

    # Truncate x-axis labels
    ax.set_xticklabels([label[:50] + '...' if len(label) > 10 else label for label in pivot_table.index], 
                       rotation=0, fontsize=16)
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.show()

plot_percentage_distribution(pivot_table)
import matplotlib.pyplot as plt

# %% ############################### State Distribution of Patients
# %% Create new dataframe addding - Dataframe
#  Create new dataframe with insurance and state information
df_med = dem.copy()
# id also like to add State from site_metadata to df_med on the FACILITY_DISPLAY_ID
df_med = df_med.merge(site_metadata[['FACILITY_DISPLAY_ID', 'State']], on='FACILITY_DISPLAY_ID', how='left')

# plot a graph of the patients health insurance of those who are using drug Exondys 51 and those who are Exondys 51 Amenable.
import matplotlib.pyplot as plt

# Create a new df_med that explodes the hltin column to get the insurance type of each patient
df_med_exploded = df_med.assign(hltin=df_med['hltin'].str.split(',')).explode('hltin')
df_med_exploded['hltin'] = df_med_exploded['hltin'].replace({
    r'.*Employer-Sponsored Disability Insurance.*': 'Employer Ins',
    r'.*Private or group health insurance.*': 'Private Ins',
    r".*Veteran's Administration Benefits / Military Insurance*": 'Veteran/Military Ins',
    r".*specify:*": 'Other',
}, regex=True)

def plot_health_insurance_distribution(df, drug, eligible, insurance_col='hltin'):
    using_drug = df[df[drug] == 1][insurance_col]
    eligible_drug = df[(df[eligible] == 1) & (df[drug] != 1)][insurance_col]

    fig, ax = plt.subplots(figsize=(12, 8))

    using_drug_counts = using_drug.value_counts()
    eligible_drug_counts = eligible_drug.value_counts()

    combined_counts = pd.DataFrame({
        f'Using {drug}': using_drug_counts,
        f'{"Amenable" if dstype == "DMD" else "Eligible"} for {drug}': eligible_drug_counts
    }).fillna(0)

    # Sort the combined counts by the total number of patients (highest to lowest)
    combined_counts = combined_counts.sort_values(by=[f'Using {drug}', f'{"Amenable" if dstype == "DMD" else "Eligible"} for {drug}'], ascending=False)

    combined_counts.plot(kind='bar', stacked=True, ax=ax, color=['blue', 'red'], alpha=0.7)

    ax.set_title(f'Health Insurance of Patients Using and {"Amenable" if dstype == "DMD" else "Eligible"} for {drug}', fontsize=24)
    ax.set_ylabel('Number of Patients', fontsize=21)
    ax.set_xlabel('', fontsize=14)  # Set x-axis label to an empty string
    ax.legend(fontsize=21)

    # Truncate x-axis labels
    ax.set_xticklabels([label[:50] + '...' if len(label) > 10 else label for label in combined_counts.index], 
                       rotation=70, fontsize=16)
    ax.tick_params(axis='y', labelsize=14)

    # Set y-axis limit to 100
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()
if dstype == 'DMD':
    plot_health_insurance_distribution(df_med_exploded, 'Exondys 51', 'Exondys 51 Amenable')
    plot_health_insurance_distribution(df_med_exploded, 'Vyondys 53', 'Vyondys 53 Amenable')
    plot_health_insurance_distribution(df_med_exploded, 'Viltepso 53', 'Viltepso 53 Amenable')
    plot_health_insurance_distribution(df_med_exploded, 'Amondys 45', 'Amondys 45 Amenable')
if dstype == 'SMA':
    plot_health_insurance_distribution(df_med_exploded, 'Spinraza', 'Spinraza Amenable')
    plot_health_insurance_distribution(df_med_exploded, 'Zolgensma', 'Zolgensma Amenable')
    plot_health_insurance_distribution(df_med_exploded, 'Evrysdi', 'Evrysdi Amenable')
if dstype == 'ALS':
    plot_health_insurance_distribution(df_med_exploded, 'Edaravone', 'Edaravone Amenable')
    plot_health_insurance_distribution(df_med_exploded, 'Riluzole', 'Riluzole Amenable')
    plot_health_insurance_distribution(df_med_exploded, 'Nuedexta', 'Nuedexta Amenable')

# %% Plotly Figure State distribution separte subplots
# "<br>Distribution of {dstype} Participants by State: On Therapeutic vs. Eligible*"
drug_pairs = list(zip(drug_use_cols, drug_amenable_cols))

from plotly.subplots import make_subplots
from collections import Counter
import plotly.graph_objects as go

fig = make_subplots(
    cols=len(drug_pairs), 
    rows=2, 
    subplot_titles=[f'{drug}' for drug, _ in drug_pairs] + [f'{amenable.replace("Amenable", "Eligible") if dstype != "DMD" else amenable}' for _, amenable in drug_pairs],
    vertical_spacing=0.1,  # Adjust vertical spacing between plots
    horizontal_spacing=0.05  # Adjust horizontal spacing between plots
)

for i, (drug, amenable) in enumerate(drug_pairs, start=1):
    # Sort states by the number of participants using the drug
    sorted_states_drug = df_med[df_med[drug] == 1]['State'].value_counts().index
    sorted_states_amenable = df_med[(df_med[amenable] == 1) & (df_med[drug] != 1)]['State'].value_counts().index

    # Add histograms for drug and amenable columns
    fig.add_trace(
        go.Histogram(
            x=pd.Categorical(df_med[df_med[drug] == 1]['State'], categories=sorted_states_drug, ordered=True),
            name=f'{drug} ({df_med[df_med[drug] == 1]["State"].count()})',
            marker_color='blue',
            showlegend=False  # Remove legend
        ),  
        row=1, col=i
    )
    fig.add_trace(
        go.Histogram(
            # Filter for amenable but not using the drug
            x=pd.Categorical(df_med[(df_med[amenable] == 1) & (df_med[drug] != 1)]['State'], categories=sorted_states_amenable, ordered=True),
            name=f'{amenable} ({df_med[(df_med[amenable] == 1) & (df_med[drug] != 1)]["State"].count()})',
            marker_color='red',
            showlegend=False  # Remove legend
        ),
        row=2, col=i
    )

# Calculate the maximum y value for each drug and amenable pair
max_y_values = []
for drug, amenable in drug_pairs:
    max_y_drug = df_med[df_med[drug] == 1]['State'].value_counts().max()
    max_y_amenable = df_med[(df_med[amenable] == 1) & (df_med[drug] != 1)]['State'].value_counts().max()
    max_y_values.append(max(max_y_drug, max_y_amenable))

# Set the y-axis limit for each subplot
for i in range(1, len(drug_pairs) + 1):
    max_y = max(max_y_values[i - 1], 50)
    fig.update_yaxes(range=[0, max_y], row=1, col=i)
    fig.update_yaxes(range=[0, max_y], row=2, col=i)

fig.update_layout(
    height=800,  # Adjust height to make the title higher
    width=1400,  # Adjust width to make the graphs wider
    title_y=0.95,  # Adjust title position
    title_font=dict(size=16),  # Set title font size to 15
    font=dict(size=10)  # Adjust font size
)
fig.update_yaxes(title_text='Participants')
fig.show()

fig.write_image(PATH_DATAEXPORT / f'{dstype}_drug_distribution_by_state.png')
# print(f"Figure saved to {PATH_DATAEXPORT / f'{dstype}_drug_distribution_by_state.png'}")

# %% Plotly for distrubtion of amenable and drug by state stacked 
# "<br>Distribution of {dstype} Participants by State: On Therapeutic vs. Eligible*"
fig = make_subplots(
    cols=len(drug_pairs), 
    rows=1, 
    subplot_titles=[f'{drug}' for drug, _ in drug_pairs],
    vertical_spacing=0.1,  # Adjust vertical spacing between plots
    horizontal_spacing=0.05  # Adjust horizontal spacing between plots
)

for i, (drug, amenable) in enumerate(drug_pairs, start=1):
    # Add overlapped bar for drug and amenable columns
    fig.add_trace(
        go.Bar(
            x=df_med['State'],
            y=df_med[df_med[drug] == 1]['State'].value_counts().reindex(df_med['State'].unique(), fill_value=0),
            name=f'{drug}',
            marker_color='blue',
            opacity=0.6
        ),  
        row=1, col=i
    )
    fig.add_trace(
        go.Bar(
            x=df_med['State'],
            y=df_med[df_med[amenable] == 1]['State'].value_counts().reindex(df_med['State'].unique(), fill_value=0),
            name=f'{amenable.replace("Amenable", "Eligible")}',
            marker_color='red',
            opacity=0.6
        ),
        row=1, col=i
    )

fig.update_layout(
    barmode='overlay',
    height=800,  # Adjust height to make the title higher
    width=1400,  # Adjust width to make the graphs wider
    title_text=f"<br>Distribution of {dstype} Participants by State: On Therapeutic vs. Eligible*",
    title_y=0.95,  # Adjust title position
    title_font=dict(size=16),  # Set title font size to 15
    font=dict(size=10)  # Adjust font size
)
fig.update_yaxes(title_text='Participants')
fig.show()

fig.write_image(PATH_DATAEXPORT / f'{dstype}_drug_distribution_by_state_overlapped.png')
# 
# %% Plotly for ALL Patients by state distribution
# "<br>Distribution of {dstype} Participants by State"
import plotly.express as px

# Create a new dataframe with the count of participants by state
state_counts = df_med['State'].value_counts().reset_index()
state_counts.columns = ['State', 'Number of Participants']

# Create a bar plot using plotly express
fig = px.bar(state_counts, x='State', y='Number of Participants', title=f'Distribution of {dstype} Participants by State')

# Update layout for better visualization
fig.update_layout(
    height=600,
    width=1000,
    title_font=dict(size=24),  # Increase title font size
    font=dict(size=18)  # Increase overall font size
)

fig.show()

fig.write_image(PATH_DATAEXPORT / f'{dstype}_all_participants_distribution_by_state.png')
# %% END