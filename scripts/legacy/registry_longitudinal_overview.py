# registry_descriptive_summary_notebook.py
# %% ################## [markdown]
# # Registry Descriptive Summary Notebook
#
# This notebook contains the code to generate a descriptive summary of the registry data. The summary includes statistics on the number of participants, encounters, and visits per disease type, as well as the average number of encounters per participant, the average time between visits, and the total time in person. The summary also includes a table of encounter bins and their counts and percentages per disease type and for all data.
#
# The summary is generated using the `registry_visits_stats` module, which contains functions to calculate statistics on registry visits data. The module includes functions to calculate the average number of encounters per participant, the average time between visits, and the total time in person. The module also includes functions to process the registry visits data, remove outliers, and create a summary table of encounter bins and their counts and percentages.
#
# The summary is displayed in a table format and includes the following information:
#
# - Total number of participants and visits per disease type
# - Average number of encounters per participant
# - Average time between visits
# - Total time in person
# - Table of encounter bins and their counts and percentages per disease type and for all data
#
# The summary is exported to an Excel file and saved in the `sharables` directory.
#
# The code is organized into the following sections:
#
# 1. Import Libraries
# 2. Data
# 3. MOVR Total
# 4. Registry Visit Stats
# 5. Summary Table
# 6. Geo Plots
# 7. Big Computation
# 8. Exports
# 9. Depricated
#

              
# %% ################## Import Libraries 
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

from path_manager import PATH_SHARABLES, PATH_DATAEXPORT
from data_manager import (config, #config, CONFIG_PATH,
                          site_metadata, #site_metadata
                          dstype, list_delements, #data dictionary excel doc as a pd.Dataframe
                          database, all_hub, all_idx,all_enc, #all data
                          data, idx,hub,dem,enc,dia,med, #main dataframes
                          dem_summary, enc_summary, #demographics and encounter summary
                        #   data_to_filter, # data + additional filtering for (age,weight) and
                          data_dict) #filtered Data Dictionary  as pd.Dataframe

from utils import plot_mod_utils, cohort_utils, time_utils
from utils.registry_stats import registry_visits_stats
from utils.registry_stats import registry_in_person_time
from utils.registry_stats import registry_geoplot

# %% ################# Data
# Prepare Data
str_dstype              = dstype
# All Disease Types Included idx and hub data saved 
df_idx_visits           = all_hub.copy() 
df_idx_visits_with_state= registry_geoplot.add_state_to_visits(df_idx_visits, site_metadata)


# Remove duplicates and use the first encounter date
df_idx_all3 = df_idx_visits_with_state.sort_values(by='encntdt').drop_duplicates(subset=['FACPATID'], keep='first')

# table of total sites per dstype and total states per dstype in df_idx_visits_with_state
table_sites_states = df_idx_visits_with_state.groupby('dstype').agg(
    Total_Sites=('FACILITY_DISPLAY_ID', 'nunique'),
    Total_States=('State', 'nunique')
).rename_axis('Disease Type')

# Calculate the total row using nunique for FACILITY_DISPLAY_ID and STATE
total_sites_states_row = pd.DataFrame({
    'Total_Sites': [df_idx_visits_with_state['FACILITY_DISPLAY_ID'].nunique()],
    'Total_States': [df_idx_visits_with_state['State'].nunique()]
}, index=['All'])

# Append the total row to the table
table_sites_states = pd.concat([table_sites_states, total_sites_states_row])

display(table_sites_states)


# Give me list of [FACPATID,CASE_ID,SCHEDULED_FORN_NAME] whose  meet the following critieria
# 1. encntdt is less before 2019-01-01
# 2. enroldt is null
usndr_df = (all_hub
            # .loc[all_hub['encntdt'] < '2019-01-01']
            .loc[all_hub['enroldt'].isnull(), 
                 ['FACPATID', 'CASE_ID', 'SCHEDULED_FORM_NAME', 
                  'encntdt', 'dstype', 'enroldt', 'Total_encounters', 
                  'dem_status', 'dia_status', 'enc_status']])

# %% ################# MOVR Total
# Group by 'dstype' and create a summary table
summary_table = df_idx_all3.groupby('dstype').agg(
    Total_Enrollments=('FACPATID', 'nunique'),
    Total_Visits=('Total_encounters', 'sum')
).rename_axis('Disease Type')

# Calculate the total row
total_row = summary_table.sum().to_frame().T
total_row.index = ['All']

# Append the total row to the summary table
summary_table = pd.concat([summary_table, total_row])

#%% Registry Visit Stats 

# Calculate total average encounter count per participant
dict_avg_enc_count, table_avg_enc_count = registry_visits_stats.calculate_average_encounter_per_participant_total(df_idx_all3)
avg_enc_count_per_dt = dict_avg_enc_count[str_dstype]['Mean']

# Process registry visits by bins
dict_visit_bins, table_visit_bins = registry_visits_stats.process_registry_visit_bins(df_idx_visits)
# Initialize an empty DataFrame to store the combined table
combined_table_visit_bins= pd.DataFrame()
for dstype, table in dict_visit_bins.items():
    # Transpose the table
    table.set_index('Encounter Bin', inplace=True)
    new_table = table['Percentage (Count)'].to_frame().T
    new_table.index = [f'{dstype}']
    
    # Append the new table as a row to the combined table
    combined_table_visit_bins = pd.concat([combined_table_visit_bins, new_table], axis=0)

# Process encounter duration by disease type
table_avg_dur_per_disease, df_avg_dur_per_patient, processed_visits = registry_visits_stats.calculate_average_encounter_duration(df_idx_visits,plot=False)

#%% Combine all tables into a single DataFrame
combined_table = pd.concat([summary_table,
                            table_avg_enc_count['Mean(IQR) Visits per Participant'],
                            table_avg_dur_per_disease['Mean(IQR) Days Between Visits'],
                            combined_table_visit_bins,
                            table_sites_states
                            ], axis=1)
# Reorder the rows based on the specified index order
index_order = ['DMD', 'SMA', 'ALS', 'BMD', 'FSHD', 'LGMD', 'Pompe', 'All']
combined_table = combined_table.reindex(index_order)
combined_table = combined_table.sort_values(by='Total_Enrollments', ascending=False)
# Ensure 'All' is the last row
combined_table = pd.concat([combined_table.drop('All'), combined_table.loc[['All']]])

# Display the combined table
display(combined_table)

# %% ################# Geo Plots

# Create a choropleth map for total encounter per state by percentage
frequency_tables, fig =registry_geoplot.create_geoplot_average_encounters(df_idx_visits, site_metadata)
fig.show()

# save the figure
fig.update_layout(
    title=dict(
        text='Average Encounters per Participant State Distribution',
        x=0.5,
        xanchor='center',
        yanchor='top'
    ),
    legend=dict(
        x=1,
        y=1,
        xanchor='right',
        yanchor='top'
    ),
    margin=dict(l=20, r=20, t=40, b=20)
)
fig.write_image(PATH_DATAEXPORT / 'average_encounters_per_participant_state_distribution.png')

# %% Create a choropleth map for total enrollment per state by percentage
frequency_tables, figures = registry_geoplot.create_geoplot_enrollment_distribution(df_idx_visits, site_metadata)

# Display the figures
for disease, fig in figures.items():
    fig.show()
    # save the figure
    fig.write_image(PATH_DATAEXPORT / f'{disease}_enrollment_distribution.png')


# %% ################# Big Computatoin
# %% Transform Clinical Visits & Calculate Time Between Visits and Total Time In Person.
dict_df_time, dict_time_metrics, table_time = registry_in_person_time.calculate_registry_stats(df_idx_visits)

# %% Format the table_time DataFrame
# Round float columns to 2 decimal places
table_time = table_time.round(2)

# Convert timedelta columns to days
# table_time['Raw in Person Time (days)'] = table_time['Raw in Person Time (days)'].dt.days
# table_time['Average Lookback Period (days)'] = table_time['Average Lookback Period (days)'].dt.days
# table_time['Total Time In Person (days)'] = table_time['Total In Person Time (days)'].astype(int)

# # Drop the 'Total In Person Time (days)' column
# table_time.drop(columns='Total In Person Time (days)', inplace=True)
# Calculate months and years for 'Average Lookback Period (days)'
table_time['Average Lookback Period (months)'] = (table_time['Average Lookback Period (days)'] / 30.44).round(1)
table_time['Average Lookback Period (years)'] = (table_time['Average Lookback Period (days)'] / 365.25).round(1)
# display
display(table_time)

# Add total 

# %% ################# Checks
# Check total time_table
dict_df_time[str_dstype].info()
# %% ################# Exports
# %% export combined_table to PATH_SHARABLES
# todays date
today = pd.Timestamp.today().strftime('%Y-%m-%d')
file_path = PATH_DATAEXPORT / f'registry_descriptive_stat_{today}.xlsx'
combined_table.to_excel(file_path,index=True)

file_path = PATH_DATAEXPORT / f'registry_time_descriptive_stat_{today}.csv'
table_time.to_excel(file_path,index=True)
# print location so that i can click on it in jupyter lab
print(f'Exported to {file_path}')
# %% ################# Depricated
# %% MOVR plot_mod_utils

# Import necessary libraries
from plotnine import scale_x_discrete, theme, element_text
# Enrollment Growth per Year
mod_plots = plot_mod_utils.Mod_Plots(str_dt_combine='All',boo_fig_save=True)
table_enrollment, df_enr, p    = mod_plots.movr_growth_per_year(df_idx_all3,'Enrollment') # df_main
p = (p 
     + scale_x_discrete(limits=df_enr['Date'].unique())
     + theme(axis_text_x=element_text(rotation=0)))
print(p)

# Encounter Growth per Year 
table_encounter , df_enc, p    = mod_plots.movr_growth_per_year(df_idx_visits,'Encounter') 
p = (p 
     + scale_x_discrete(limits=df_enc['Date'].unique())
     + theme(axis_text_x=element_text(rotation=0)))
print(p)

# Display the tables of encounter bins and counts
avg_enc_count_per_dt, table_bin_enc, p = mod_plots.hist_enc_count(df_idx_visits)
print(p)
