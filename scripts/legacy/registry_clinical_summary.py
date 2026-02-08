# registry_demographic_summary.py
# %% [markdown]
# Registry Clinical Summary
#
# This notebook provides a summary of the clinical data in the registry. The data is summarized
# by disease type (`dstype`) and includes the following information:
#
# - Total number of participants
# - Total number of sites
# - Total average encounters

# Standard Library Imports
from sys import path
from pathlib import Path
import pandas as pd
from IPython.display import display

# Get the project root from the environment variable
project_root = Path(__file__).resolve().parents[3]
path.append(str(project_root))  # type: ignore

# Import setup_project_paths from setup_paths.py
from core.setup_paths import setup_project_paths
setup_project_paths()

from path_manager import PATH_SHARABLES, PATH_DATAEXPORT
from data_manager import (config, #config, CONFIG_PATH,
                          site_metadata, #site_metadata
                          movr_dict,all_data, #all_data
                          dstype, list_delements, #data dictionary excel doc as a pd.Dataframe
                          database, all_hub, all_idx,all_enc, #all data,
                          dia_summary, dem_summary, enc_summary, #summary
                          data, hub, dia, enc, dem, med, idx, #disease specific data,
)

from utils.data_processing import summary_utils
from utils.data_dictionary_utils import rename_columns, save_enc_renamed_info
from utils.transformations import transform_demographics

print(f"\n Disease Type (dstype): {dstype}")


# %% ############################################################################# Demographic Summary Statistics
# %% Plotting defintion
import matplotlib.pyplot as plt

###Set Parameters 
mda_blue = '#485CC7'
mda_gold = '#f1b434'
mda_red = '#E64560'
mda_green = '#5DD9C2'
mda_purple = '#4A327D'
mda_tan = '#EBE4E4'
snow4 = '#8B8989'
sgigray76 = '#C1C1C1'
sgigray16 = '#282828'
mistyrose1 = '#FFE4E1'
mintcream = '#F5FFFA'
# Plotting defintion
def plot_pie_chart(data, column, title, export_path):
    counts = data[column].value_counts()
    plt.figure(figsize=(8, 8))
    
    # Define the colors
    colors = [mda_blue, mda_gold, mda_red, mda_green, mda_purple, mda_tan, snow4, sgigray76, sgigray16, mistyrose1, mintcream]
    
    # Plot the pie chart
    plt.pie(counts, labels=['']*len(counts), autopct='', startangle=140, colors=colors[:len(counts)])
    plt.title(title, fontsize=32)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()

    # Create legend with truncated percentages at the front
    legend_labels = [f'{count / counts.sum() * 100:.1f}% {label[:40]}' for label, count in counts.items()]
    plt.legend(legend_labels, title=column.replace('_', ' ').title(), loc="best", fontsize=14)

    # Save the pie chart
    # plt.savefig(export_path, bbox_inches='tight')
# %% Demographic & IDX & Diagnosis

temp_dem = dem.copy()
# Demographic Preprocessing
dem_1 = transform_demographics.clean_race_column(temp_dem, 'ethnic')
dem_1['hltin'] = dem_1['hltin'].astype(str)
# dem_1['hltin'] = dem_1['hltin'].replace('Not Reported', pd.NA)

def clean_health_insurance_column(df, column):
    """
    Clean the 'Type of Health Insurance' column by splitting entries, exploding the dataframe,
    and filling NaN values with 'Unknown'. Additionally, delete rows that contain 'specify:' in the column.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the column to clean.
    column (str): The name of the column to clean.
    
    Returns:
    pd.DataFrame: The cleaned dataframe with the specified column processed.
    """
    # Split the entries by comma and explode the dataframe
    df[column] = df[column].str.split(',')
    df = df.explode(column)
    
    # Fill NaN with 'Unknown' if no insurance is reported
    df[column] = df[column].fillna('Unknown')
    
    # Delete rows that contain 'specify:' in the column
    df = df[~df[column].str.contains('specify:', case=False, na=False)]
    
    return df

# Apply the cleaning function to the 'Type of Health Insurance' column
dem_hltin_explode = clean_health_insurance_column(temp_dem, 'hltin')

dem_1['ethnic'] = dem_1['ethnic_new']
columns_to_drop = [
    'CASE_ID','PATIENT_DISPLAY_ID','SCHEDULED_FORM_NAME',
    'FACPATID',
    # 'FACILITY_DISPLAY_ID',
    'enroldt','enroldt.P','dob','dob.P','ethnicosp','hltinosp','has_mname','dob1','dob1.P','sex',
    'nonmdapc','inschool','inschyes','edulvl','edulvl1','edulvl2','employ',
    'usndr','Source_sheet','Demographics_MainData',
    'ethnic_new'
]
dem_1 = dem_1.rename(columns={'ethnic':'Race','ethnicity':'Ethnicity','gender':'Gender','hltin':'Type of Health Insurance'})
dem_1['Race'] = dem_1['Race'].replace('0', 'Multi racial')

# Check if columns exist in the dataframe before dropping
columns_to_drop = [col for col in columns_to_drop if col in dem_1.columns]

dem_dropped = dem_1.drop(columns=columns_to_drop)
dem_renamed2 = rename_columns(dem_dropped, movr_dict)


renamed_dem_columns = {
    'FACPATID':'Total Enrollment',
    'FACILITY_DISPLAY_ID':'Site',
    'Disease Type':'Enrolled Participants',
}

# Create a summary table of the demographic data
dem_summary_stats = summary_utils.create_describe_summary_table(dem_renamed2,rename_dict=renamed_dem_columns)
dem_summary_stats = summary_utils.add_value_counts_to_summary(dem_renamed2, dem_summary_stats,'Race')
dem_summary_stats = summary_utils.add_value_counts_to_summary(dem_renamed2, dem_summary_stats,'Ethnicity')


print(f"Demographic Summary Statistics for {dstype}")
display(dem_summary_stats)

# Plot 
plot_pie_chart(dem_1, 'Race', f'{dstype} Race', PATH_DATAEXPORT / f'{dstype}_RACE_chart.png')
plot_pie_chart(dem_1, 'Ethnicity', f'{dstype} Ethnicity', PATH_DATAEXPORT / f'{dstype}_ETH.png')

plot_pie_chart(dem_hltin_explode, 'hltin', f'{dstype} Health Insurance', PATH_DATAEXPORT / f'{dstype}_health_insurance_pie_chart.png')



# %% IDX clean

# Calculate Age new that includes idx['dob'] and calculates to today's date but ignores those that have a 'dis_status' other than 'DNE' value
idx['Age'] = (pd.Timestamp.today() - idx['dob']).dt.days / 365.25
idx['Age'] = idx.apply(lambda row: (min(pd.Timestamp.today(), row['deathdt'], row['stuexdt']) - row['dob']).days / 365.25 if row['dis_status'] != 'DNE' else (pd.Timestamp.today() - row['dob']).days / 365.25, axis=1)
idx['Age_at_diagnosis'] = idx['Age_at_diagnosis'].apply(lambda x: max(x, 0))
idx['Age_at_enrollment']= idx['Age_at_enrollment'].apply(lambda x: max(x, 0))
idx['dis_status'] = idx['dis_status'].replace('DNE', 'No discontinuation record')

idx_select = idx[[
    # 'FACPATID','FACILITY_DISPLAY_ID',
                  'Age_at_enrollment','Age_at_diagnosis',
                #   'Age', 'Age_at_diagnosis',                  # need to update to be Age_last_encounter'
                  'Total_encounters',
                  'dis_status',]]

renamed_idx_columns = {
    'FACPATID':'Total Enrollment',
    'FACILITY_DISPLAY_ID':'Site',
    'Disease Type':'Enrolled Participants',
    'dis_status':'Discontinued Status',
    'Total_encounters':'Number of Study Visits',
}
idx_summary_stats = summary_utils.create_describe_summary_table(idx_select,rename_dict=renamed_idx_columns)

# print the Derived Summary Statistics (idx dataframe)
print(f"IDX Summary Statistics for {dstype}")
display(idx_summary_stats)

# %% Demo and idx combined Summary Stats
# Combine the demographic and idx summary statistics
summary_stats = pd.concat([dem_summary_stats, idx_summary_stats], axis=0)

# Reorder the summary_stats so that index starts with ['Enrolled Participants','Number of Study Visits','Site']
index_order = ['Enrolled Participants','Site']
remaining_indices = [idx for idx in summary_stats.index if idx not in index_order]
summary_stats = summary_stats.reindex(index_order + remaining_indices)
summary_stats[['Mean', 'SD','Median']] = summary_stats[['Mean', 'SD','Median']].astype(float).round(1)
summary_stats = summary_stats.replace('nan', '')

# Print the combined summary statistics
# print(f"Combined Summary Statistics for {dstype}")
# display(summary_stats)

# %% Age at Enrollment Histogram
import matplotlib.pyplot as plt

ylmit = 50
if dstype == 'ALS':
    ylmit = 250
if dstype == 'SMA':
    ylmit = 500
if dstype == 'DMD':
    ylmit = 250

if dstype in ['FSHD','LGMD','Pompe']:
    ylmit = 50

# idx['Age_at_enrollment'] Plot 
idx['Age_at_enrollment'].plot.hist(bins=20, alpha=0.5, color=mda_blue, edgecolor='black', linewidth=1.2, label='Age at Enrollment')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Age at Enrollment', fontsize=16)
plt.ylabel(f'Number of {dstype} Participants', fontsize=16)
plt.ylim(0, ylmit)
plt.legend(fontsize=14)
plt.tight_layout(pad=2)

# save the plot 
# plt.savefig(PATH_DATAEXPORT / f'{dstype}_registry_age_at_enrollment_histogram.png', bbox_inches='tight')
print(f'{dstype} Age at Enrollment Histogram saved to {PATH_DATAEXPORT / f"{dstype}_registry_age_at_enrollment_histogram.png"}')

# idx['Age_at_diagnosis'] Plot
idx['Age_at_diagnosis'].plot.hist(bins=20, alpha=0.5, color=mda_gold, edgecolor='black', linewidth=1.2, label='Age at Diagnosis')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Age', fontsize=16)
plt.ylabel(f'Number of {dstype} Participants', fontsize=16)
plt.ylim(0, ylmit)
plt.legend(fontsize=14)
plt.tight_layout(pad=2)

if dstype in ['ALS', 'SMA']:
    # Age at system onset plot
    column_name = 'alsdgnag' if dstype == 'ALS' else 'smadgnag'
    # fix dia colum name by changine negative numbers to 0 
    dia[column_name] = dia[column_name].apply(lambda x: max(x, 0))

    dia[column_name].plot.hist(bins=20, alpha=0.5, color=mda_red, edgecolor='black', linewidth=1.2, label='Age at System Onset')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Age', fontsize=16)
    plt.ylabel(f'Number of {dstype} Participants', fontsize=16)
    plt.ylim(0, ylmit)
    plt.legend(fontsize=14)
    plt.tight_layout(pad=2)

if dstype in ['SMA']:
    # Adjust smaclass value_counts
    dia['smaclass'] = dia['smaclass'].apply(lambda x: x if 'Type' in str(x) else 'Other')
    dia['smaclass'] = dia['smaclass'].apply(lambda x: x[:12] if isinstance(x, str) else x)

    # Plot age of onset in subplots grouped by smaclass
    unique_smaclass = dia['smaclass'].unique()
    num_plots = len(unique_smaclass)
    fig, axes = plt.subplots((num_plots + 1) // 2, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    for ax, smaclass in zip(axes, unique_smaclass):
        subset = dia[dia['smaclass'] == smaclass]
        subset[column_name].plot.hist(bins=20, alpha=0.5, color=mda_blue, edgecolor='black', linewidth=1.2, ax=ax)
        ax.set_title(f'Age at System Onset ({smaclass})', fontsize=16)
        ax.set_ylabel(f'Number of {dstype} Participants', fontsize=14)
        ax.legend([f'Age at System Onset ({smaclass}) (n={subset[column_name].count()})'], fontsize=12)
        # ax.grid(True)
        # ax.set_yticks(range(0, 100,10))
    plt.xlabel('Age', fontsize=16)
    plt.tight_layout(pad=2)
   

# save the plot 
# plt.savefig(PATH_DATAEXPORT / f'{dstype}_registry_age_at_diagnosis_histogram.png', bbox_inches='tight')
print(f'{dstype} Age at Histogram saved to {PATH_DATAEXPORT / f"{dstype}_registry_age_at_diagnosis_histogram.png"}')

# %% ############################################################################# Diagnosis Summary Statistics
# %% Diagnosis Preprocessing and Summary Statistics
# Clean the 'bdypt' column by splitting entries, exploding the dataframe,
# and filling NaN values with 'Unknown'. Additionally, delete rows that contain 'specify:' in the column.

if dstype == 'DMD':
    plot_pie_chart(dia,'dmdgntcf','DMD Genetic Confirmation', PATH_DATAEXPORT / f'{dstype}_genetic_confirmation_pie_chart.png')

if dstype == 'ALS':
    # def clean_body_part_column(df, column):
    #     df[column] = df[column].str.split(',')
    #     df = df.explode(column)
    #     df[column] = df[column].fillna('Unknown')
    #     df = df[~df[column].str.contains('specify:', case=False, na=False)]
    # return df

    # # Apply the cleaning function to the 'bdypt' column
    # dia_cleaned = clean_body_part_column(dia, 'bdypt')
    # Plot and save pie chart for 'bdypt' column
    plot_pie_chart(dia_cleaned, 'bdypt', f'{dstype} Body Region First Affected', PATH_DATAEXPORT / f'{dstype}_body_part_pie_chart.png')
    plot_pie_chart(dia,'genemut', f'{dstype} Genetic Mutation', PATH_DATAEXPORT / f'{dstype}_genetic_mutation_pie_chart.png')

if dstype == 'SMA':
    # Plot and save pie chart for 'sma_type' column
    # pie charge smadgmad 'How diagnosis initally made
    plot_pie_chart(dia,'smadgmad', f'{dstype} Diagnosis Method', PATH_DATAEXPORT / f'{dstype}_diagnosis_method_pie_chart.png')
    plot_pie_chart(dia,'smaclass', f'{dstype} Class', PATH_DATAEXPORT / f'{dstype}_sma_class_pie_chart.png')
    plot_pie_chart(dia,'smadgcnf', f'{dstype} Diagnosis Confirmation', PATH_DATAEXPORT / f'{dstype}_diagnosis_confirmation_pie_chart.png')
    plot_pie_chart(dia,'smn1cn', f'{dstype} SMN1 Copy Number', PATH_DATAEXPORT / f'{dstype}_smn1_copy_number_pie_chart.png')
    # plot_pie_chart(dia, 'sma_type', f'{dstype} Type', PATH_DATAEXPORT / f'{dstype}_type_pie_chart.png')
    # plot_pie_chart(dia, 'genemut', f'{dstype} Genetic Mutation', PATH_DATAEXPORT / f'{dstype}_genetic_mutation_pie_chart.png')
# %% ############################################################################# Clinical Summary Statistics
# %% Clinical Preprocessing 
# Clinical Summary onto final encounter data
enc_last = enc.sort_values('encntdt').drop_duplicates('FACPATID', keep='last')
enc_last_renamed = rename_columns(enc_last, movr_dict)
enc_last_summary = save_enc_renamed_info(enc_last_renamed, enc_last)

if dstype == 'ALS':
    enc_select = enc_last[[
        'apprxfall', 'hospvis', 'amblloss', 'spchloss', 'gstrmy', 'fstniv', 'trach', 'nonivventyn', 'trachventyn', 
        'alsfrsr','admmthd', 'alsfrstl', 'curramb', 'asstdvc', 'pulmdvc', 'curmed2', 'pftfvc', 'fvcrslt'
    ]]

if dstype == 'SMA':
    enc_select = enc_last[[
        'curramb','smafunc', 'clmbstrs', 'rollcmp', 'stsup', 'stunsp', 'stdsprt', 'stdunspt', 'walkspt', 'walkuspt', 
        'hfmsesc', 'nonivventyn', 'trachventyn', 'dmthrpy', 'sptrtdatld1', 'spindosld1', 'spiroutld1', 'spuncirld1'
    ]]

if dstype == 'DMD':
    enc_select = enc_last[[
        'ttrsupn', 'ttcstr', 'ttwr10m', 'poul2', 'armshldr', 'hipslegs', 'whlchr', 
        'surgvis2', 'hospvis2', 'curramb', 'bhvdst', 'glcouse','glcdoseauto','stfract','cushng','wtgain','pubd','grth','hyprtn','bhvdst',
          'pftest', 'fev1rslt', 'fvcpctpd'
    ]]
    field_name_add  ='*Has the patient had any surgeries since their last visit? '
    field_name_add2 ='*Has the patient had any hospitalizations since their last visit?'
    field_name_add3 ='*Does the Patient currently use Glucocorticoids?'
    field_name_add4 = '*Arms and Shoulders:'
    field_name_add5 = '*Hips and Legs:'


if dstype in ['ALS','SMA']:
    enc_renamed2 = rename_columns(enc_select, movr_dict)
    enc_summary_stats = summary_utils.create_describe_summary_table(enc_renamed2)
    # display(enc_summary_stats)

    if dstype == 'ALS':
        #plot the distrubtion of the ALS FRS Score  'alsfrstl' from enc_last
        enc_last['alsfrstl'].plot.hist(bins=20, alpha=0.5, color=mda_blue, edgecolor='black', linewidth=1.2, label='ALS FRS Score')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('ALS FRS Score', fontsize=16)
        plt.ylabel(f'Number of {dstype} Participants', fontsize=16)
        plt.legend(fontsize=14)
        plt.tight_layout(pad=2)

    plot_pie_chart(enc_last,'clntrlyn', f'{dstype} Clinical Trial Participation', PATH_DATAEXPORT / f'{dstype}_clinical_trial_pie_chart.png')
    plot_pie_chart(enc_last,'prvclntr', f'{dstype} Previous Clinical Trial Participation', PATH_DATAEXPORT / f'{dstype}_previous_clinical_trial_pie_chart.png')
    # enc_summary_stats.to_excel(PATH_DATAEXPORT / f'{dstype}_registry_clinical_summary.xlsx',index=True)
if dstype in ['DMD']:
    enc_renamed2 = rename_columns(enc_select, movr_dict)
    enc_summary_stats = summary_utils.create_describe_summary_table(enc_renamed2)
    enc_summary_stats = summary_utils.add_value_counts_to_summary(enc_renamed2, enc_summary_stats, field_name_add)
    enc_summary_stats = summary_utils.add_value_counts_to_summary(enc_renamed2, enc_summary_stats, field_name_add2)
    enc_summary_stats = summary_utils.add_value_counts_to_summary(enc_renamed2, enc_summary_stats, field_name_add3)
    enc_summary_stats = summary_utils.add_value_counts_to_summary(enc_renamed2, enc_summary_stats, field_name_add4)
    enc_summary_stats = summary_utils.add_value_counts_to_summary(enc_renamed2, enc_summary_stats, field_name_add5)
    
    print(f"Clinical Encounter Summary Statistics for {dstype}")
    display(enc_summary_stats)

    # Clean 'armshldr' and 'hipslegs' columns in enc_last
    enc_select['armshldr'] = enc_select['armshldr'].replace(['Not tested', 'Not Tested'], 'Not Tested')
    enc_select['hipslegs'] = enc_select['hipslegs'].replace(['Not tested', 'Not Tested'], 'Not Tested')

    # Plot and save pie charts for 'glcouse', 'armshldr', and 'hipslegs' columns
    plot_pie_chart(enc_select, 'glcouse', f'{dstype} Glucocorticoid Use Distribution', PATH_DATAEXPORT / f'{dstype}_glucocorticoid_use_pie_chart.png')
    plot_pie_chart(enc_select, 'armshldr', f'{dstype} Arms and Shoulders Distribution', PATH_DATAEXPORT / f'{dstype}_arms_and_shoulders_pie_chart.png')
    plot_pie_chart(enc_select, 'hipslegs', f'{dstype} Hips and Legs Distribution', PATH_DATAEXPORT / f'{dstype}_hips_and_legs_pie_chart.png')
    plot_pie_chart(enc_select,'surgvis2', f'{dstype} Surgeries Distribution', PATH_DATAEXPORT / f'{dstype}_surgeries_pie_chart.png')
    plot_pie_chart(enc_select,'hospvis2', f'{dstype} Hospitalizations Distribution', PATH_DATAEXPORT / f'{dstype}_hospitalizations_pie_chart.png')
    plot_pie_chart(enc_last, 'crdmyo', f'{dstype} Cardiomyopathy', PATH_DATAEXPORT / f'{dstype}_cardiomyopathy_pie_chart.png')
    # # plot the distribution of glcdoseauto using a scatter plot
    # plt.figure()
    # plt.scatter(enc_last.index, enc_last['glcdoseauto'], color=mda_blue, edgecolor='black')
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel('Participants', fontsize=16)
    # plt.ylabel('Glucocorticoid Dose', fontsize=16)
    # plt.tight_layout(pad=2)

    # enc_summary_stats.to_excel(PATH_DATAEXPORT / f'{dstype}_registry_clinical_summary.xlsx',index=True)
else:
    print(f'Clinical Summary Statistics not available for this disease type: {dstype}')


# %% Repeat Group Statistics

# plot surgeries and hospitalizations
surg = data['Log_Surgery']
surg = surg[surg['FACPATID'].isin(dem['FACPATID'])]
hosp = data['Log_Hospitalization']
hosp = hosp[hosp['FACPATID'].isin(dem['FACPATID'])]
pul  = data['Log_PulmonaryDevice']
pul = pul[pul['FACPATID'].isin(dem['FACPATID'])]
asd = data['Log_AssistiveDevice']
asd = asd[asd['FACPATID'].isin(dem['FACPATID'])]

med = pd.read_parquet(project_root.parent / "Combo_Drugs.parquet")
# Filter medications to include only those participants in the demographic data
med = med[med['FACPATID'].isin(dem['FACPATID'])]

# Update 'StandardName' to group less frequent medications into 'Other'
top_medications = med['StandardName'].value_counts().nlargest(10).index
med['StandardName'] = med['StandardName'].apply(lambda x: x if x in top_medications else 'Other')

# Update 'pulmonary_device' to group less frequent devices into 'Other'
pul['pulmonary_device'] = pul['nivent'].fillna(pul['respdev'])




plot_pie_chart(surg, 'srgtype', f'{dstype} Surgeries Distribution', PATH_DATAEXPORT / f'{dstype}_surgeries_pie_chart.png')
plot_pie_chart(hosp, 'hsprsn1', f'{dstype} Hospitalizations Distribution', PATH_DATAEXPORT / f'{dstype}_hospitalizations_pie_chart.png')
plot_pie_chart(med, 'StandardName', f'{dstype} Medications Distribution', PATH_DATAEXPORT / f'{dstype}_medications_pie_chart.png')
plot_pie_chart(pul, 'nivent', f'{dstype} Pulmonary (NIV) Devices Distribution', PATH_DATAEXPORT / f'{dstype}_pulmonary_devices_pie_chart.png')
plot_pie_chart(asd,'mobdev', f'{dstype} Assistive (Mobility) Devices Distribution', PATH_DATAEXPORT / f'{dstype}_assistive_devices_pie_chart.png')
# %% Repeat Group Statistics
# Repeat Group Statistics
dict_for_repeat = {}
for key, item in data.items():
    if isinstance(item, pd.DataFrame) and key not in ['movr_config_dict', 'data_dict', 'idx', 'hub']:
        tups = item.shape
        # add another to the tuple for number nunique() for FACPATID
        tups = tups + (item['FACPATID'].nunique(),)
        dict_for_repeat[key] = tups

# Create a DataFrame from the dictionary
repeat_group_stats = pd.DataFrame(dict_for_repeat, index=['Number of Records', 'Number of Fields','Number of Unique Participants']).T

# Drop rows where 'Number of Records' is 0
# repeat_group_stats = repeat_group_stats[repeat_group_stats['Number of Records'] != 0]

# Rename index meds and call it Medications Table for clarity
repeat_group_stats = repeat_group_stats.rename(index={'meds':'Medications Table'})
# drop Encounter_Medication and Log_Medication
repeat_group_stats = repeat_group_stats.drop(index=['Encounter_Medication','Log_Medication'])

# Creat Hospitilizations from Encounter_Hospitalizations and Log_Hospitalizations and then drop those two 
repeat_group_stats.loc['Hospitalizations Table'] = repeat_group_stats.loc['Encounter_Hospitalization'] + repeat_group_stats.loc['Log_Hospitalization']
# add the number of fields from Log_Hospitalization to new Hospitalizations Table
repeat_group_stats.loc['Hospitalizations Table', 'Number of Fields'] = repeat_group_stats.loc['Log_Hospitalization', 'Number of Fields']
repeat_group_stats.loc['Hospitalizations Table', 'Number of Unique Participants'] = pd.concat([
    data['Encounter_Hospitalization']['FACPATID'], 
    data['Log_Hospitalization']['FACPATID']
]).nunique()
repeat_group_stats = repeat_group_stats.drop(index=['Encounter_Hospitalization','Log_Hospitalization'])

# Create Assistive Devices from Encounter_AssistiveDevice and Log_AssistiveDevice and then drop those two
repeat_group_stats.loc['Assistive Devices Table'] = repeat_group_stats.loc['Encounter_AssistiveDevice'] + repeat_group_stats.loc['Log_AssistiveDevice']
repeat_group_stats.loc['Assistive Devices Table', 'Number of Fields'] = repeat_group_stats.loc['Log_AssistiveDevice', 'Number of Fields']
repeat_group_stats.loc['Assistive Devices Table', 'Number of Unique Participants'] = pd.concat([
    data['Encounter_AssistiveDevice']['FACPATID'], 
    data['Log_AssistiveDevice']['FACPATID']
]).nunique()
repeat_group_stats = repeat_group_stats.drop(index=['Encounter_AssistiveDevice','Log_AssistiveDevice'])

# Create PulmonaryDevice Table from Encounter_PulmonaryDevice and Log_PulmonaryDevice and then drop those two
repeat_group_stats.loc['Pulmonary Devices Table'] = repeat_group_stats.loc['Encounter_PulmonaryDevice'] + repeat_group_stats.loc['Log_PulmonaryDevice']
repeat_group_stats.loc['Pulmonary Devices Table', 'Number of Fields'] = repeat_group_stats.loc['Log_PulmonaryDevice', 'Number of Fields']
repeat_group_stats.loc['Pulmonary Devices Table', 'Number of Unique Participants'] = pd.concat([
    data['Encounter_PulmonaryDevice']['FACPATID'], 
    data['Log_PulmonaryDevice']['FACPATID']
]).nunique()
repeat_group_stats = repeat_group_stats.drop(index=['Encounter_PulmonaryDevice','Log_PulmonaryDevice'])


if dstype == 'ALS':
    repeat_group_stats = repeat_group_stats.rename(index={'Log_Surgery': 'Surgery Table'})
else:
    # Create Surgery Table from many Encounter (Encounter_FeedingTubePlacement, Encounter_ICDPlacement, Encounter_ScoliosisSurgery, Encounter_SurgeryOther, Encounter_TendonReleaseSurgery, Encounter_Tracheostomy)
    #  and one Log_Surgery  ()
    repeat_group_stats.loc['Surgery Table'] = sum(repeat_group_stats.loc[col] for col in [
        'Encounter_FeedingTubePlacement', 'Encounter_ICDPlacement', 'Encounter_ScoliosisSurgery', 
        'Encounter_SurgeryOther', 'Encounter_TendonReleaseSurgery', 'Encounter_Tracheostomy', 'Log_Surgery'
    ] if col in repeat_group_stats.index)
    repeat_group_stats.loc['Surgery Table', 'Number of Fields'] = repeat_group_stats.loc['Log_Surgery', 'Number of Fields']
    surgery_columns = [
        'Encounter_FeedingTubePlacement', 'Encounter_ICDPlacement', 'Encounter_ScoliosisSurgery', 
        'Encounter_SurgeryOther', 'Encounter_TendonReleaseSurgery', 'Encounter_Tracheostomy', 'Log_Surgery'
    ]

    # Check if columns exist before concatenating and dropping
    existing_surgery_columns = [col for col in surgery_columns if col in data]

    if existing_surgery_columns:
        repeat_group_stats.loc['Surgery Table', 'Number of Unique Participants'] = pd.concat([
            data[col]['FACPATID'] for col in existing_surgery_columns
        ]).nunique()
        repeat_group_stats = repeat_group_stats.drop(index=existing_surgery_columns)

# Drop rows where 'Number of Records' is 0
repeat_group_stats = repeat_group_stats[repeat_group_stats['Number of Records'] != 0]

# Print the Repeat Group Statistics
# print(f"Repeat Group Statistics for {dstype}")
# display(repeat_group_stats)

# Save
# repeat_group_stats.to_excel(PATH_DATAEXPORT / f'{dstype}_registry_repeat_group_stats.xlsx',index=True)
# print(f'{dstype} Repeat Group Statistics saved to {PATH_DATAEXPORT / f"{dstype}_registry_repeat_group_stats.xlsx"}')

# %% ############################################################################# Export
# Export the summary statistics to a CSV file

# summary_stats.to_excel(PATH_DATAEXPORT / f'{dstype}_registry_demographic_summary.xlsx',index=True)
# dia_summary_stats.to_excel(PATH_DATAEXPORT / f'{dstype}_registry_diagnosis_summary.xlsx',index=True)


# %% END