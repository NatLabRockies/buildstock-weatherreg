# IMPORTS
# Import libraries
import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime as dt
import os
import json
import shutil
import subprocess
from buildstock_query import BuildStockQuery
import time
from urllib.error import HTTPError
import traceback
import re
import random

# Set environment variable to disable OneDNN prior to importing tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
# Set random seed for reproducibility
tf.random.set_seed(42)

script_start_time = dt.datetime.now()
print('Script start time:', script_start_time)

# Force program to sleep for a random amount of time between 0 and 300 seconds
# Prevents AWS token errors when multiple jobs are run simultaneously
time.sleep(random.uniform(0, 30))

# Import command line arguments
start_index = int(sys.argv[1])
end_index = int(sys.argv[2])
meta_path = sys.argv[3]
upgrade = sys.argv[4]
prefix = sys.argv[5]
output_dir = sys.argv[6]
script_dir = sys.argv[7]
counties_str = sys.argv[8]

print('start_index:', start_index)
print('end_index:', end_index)
print('meta_path:', meta_path)
print('upgrade:', upgrade)
print('prefix:', prefix)
print('output_dir:', output_dir)
print('script_dir:', script_dir)
print('counties_str:', counties_str)

print('Script to rerun this file with the same arguments:')
print(f'sbatch --job-name=chunk_{prefix}{upgrade}_{start_index}-{end_index} '
      f'./C_run_bldg_chunk_agg.sh {start_index} {end_index} {meta_path} '
      f'{upgrade} {prefix} {output_dir} {script_dir} {counties_str}')

# Import switches #TODO: Only import necessary for this script & reorder
with open(os.path.join(output_dir, 'inputs', 'switches_agg.json'), 'r') as f:
    switch = json.load(f)
## Switch that designates comstock or resstock data
sw_comstock = switch['comstock'] # if `False`, then resstock
sw_savings_shape = switch['savings_shape'] # if `False`, aggregate_timeseries
applied_only = switch['applied_only'] # if `True`, only buildings with upgrade applied
## Columns to group by; Note: if this changes, language in the 
## `process_chunk_agg` function will need to be updated following the AWS call
bsq_cols =  switch['com_bsq_cols'] if sw_comstock else switch['res_bsq_cols']
## Note: resstock upgrades do not correspond to the same # as comstock
upgrades = switch['upgrades'] # default: comstock = [0, 1, 18], resstock = [0, 1, 5]
## Number of buildings to pull per upgrade
n_bldngs = switch['n_bldngs'] # 'all' for all buildings, 'assign' for assigned building id list from csv
base_year = switch['base_year'] # Base year for the building stock
target_year = switch['target_year'] # Target year for the building stock
sw_apply_regression = switch['apply_regression']
sw_test_base = switch['test_base']
sw_save_metrics = switch['save_metrics']
sw_show_fit = switch['show_fit']
sw_save_fit = switch['save_fit']
sw_test_target = switch['test_target']
sw_cross_val = switch['cross_val']  # If True, perform cross-validation, fitting accuracy will be improved and runtime will be longer
sw_hybrid_model = switch['hybrid_model']  # If True, use a hybrid model of random forest and polynomial regression to overcome the limitations of random forest extrapolation, and runtime will be longer. If False, use only random forest.
sw_mode = switch['mode'] # Choose HVAC electricity usage, "heat_and_cool" for all HVAC (default), "heat" for heating only, "cool" for cooling only
comstock_year, comstock_release = switch['version_comstock'][0], switch['version_comstock'][1]
resstock_year, resstock_release = switch['version_resstock'][0], switch['version_resstock'][1]
lag_hours = switch['lag_hours_temperature']   # Lagged features for the dry bulb temperature to include the load inertia
base_run = switch['base_run'] # The base run type for the BuildStockQuery object
target_run = switch['target_run'] # The target run type for the BuildStockQuery object
run_types = switch['run_types'] # Run types for the BuildStockQuery object
url_base = switch['url_base']


# FUNCTIONS
def query_execution(query, my_run, retries=5, delay=10):
    """
    Helper function to retry query execution for transient failures.

    Parameters:
    query (str): The query string to execute.
    my_run (BuildStockQuery): The BuildStockQuery object.
    retries (int): Number of retry attempts.
    delay (int): Delay in seconds between retries.

    Returns:
    DataFrame: The query results.
    """
    for attempt in range(retries):
        try:
            print(f"Executing query, attempt {attempt + 1}...")
            return my_run.execute(query)
        except Exception as e:
            print(f"Query execution failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise  # Re-raise the exception if out of retries

def process_chunk_agg(run_type, upgrade, counties, bsq_cols, sw_comstock,
                      chunk_states, sw_savings_shape, df_meta, applied_only):
    """
    This function aggregates timeseries data for a specific run type, upgrade,
    enduse, and set of counties.
    It then processes the aggregated data to calculate the 'HVAC.elec' column
    and returns the DataFrame.

    Parameters:
    run_type (str): The type of run to process.
    upgrade (int): The upgrade ID to process.
    counties (list): The counties to process.
    bsq_cols (list): The columns to group by when aggregating.
    sw_comstock (bool): Whether the data is from ComStock (True) or ResStock.
    chunk_states (list): The states to process.
    sw_savings_shape (bool): Method - savings_shape or aggregate_timeseries.
    df_meta (DataFrame): The metadata DataFrame.
    applied_only (bool): If True, only buildings with upgrade applied are used.

    Returns:
    ts_agg (DataFrame): Aggregated timeseries HVAC electricity.
    """
    # TODO: Make `is_oedi/else` sections into one section each if 2 aws calls
    # Copy variables to avoid modifying the variables in the main script
    aws_cols = bsq_cols.copy()
    aws_counties = counties.copy()
    aws_run_type = run_types[run_type].copy()
    aws_upgrade = upgrade

    # Check if the run type is OEDI
    is_oedi = ('db_schema' in aws_run_type 
               and 'oedi' in aws_run_type['db_schema'])

    # Set the relevant variables based on the run type
    heating_fuel = (['NaturalGas', 'Electricity'] if sw_comstock else
                    ['Natural Gas', 'Electricity'])
    if is_oedi:
        if sw_comstock:
            elec_enduse = [
                'out.electricity.heating.energy_consumption',
                'out.electricity.cooling.energy_consumption',
                'out.electricity.fans.energy_consumption',
                'out.electricity.heat_recovery.energy_consumption',
                'out.electricity.heat_rejection.energy_consumption',
                'out.electricity.pumps.energy_consumption'
            ]
        else:
            elec_enduse = [
                'out.electricity.heating.energy_consumption',
                'out.electricity.heating_fans_pumps.energy_consumption',
                'out.electricity.heating_hp_bkup.energy_consumption',
                'out.electricity.heating_hp_bkup_fa.energy_consumption',
                'out.electricity.cooling.energy_consumption',
                'out.electricity.cooling_fans_pumps.energy_consumption'
            ]
        natural_gas = ['out.natural_gas.heating.energy_consumption']
        restrict_county = ('in.nhgis_county_gisjoin' if sw_comstock else 
                           'in.county')
        restrict_bldg_type = ('in.comstock_building_type' if sw_comstock else
                              'in.geometry_building_type_recs')
        restrict_heating_fuel = 'in.heating_fuel'

    else:
        elec_enduse = [
            'end_use__electricity__heating__kwh',
            'end_use__electricity__heating_fans_pumps__kwh',
            'end_use__electricity__heating_heat_pump_backup__kwh',
            'end_use__electricity__heating_heat_pump_backup_fans_pumps__kwh',
            'end_use__electricity__cooling__kwh',
            'end_use__electricity__cooling_fans_pumps__kwh'
        ]
        natural_gas = ['end_use__natural_gas__heating__kbtu']

        restrict_county = 'build_existing_model.county'
        restrict_bldg_type = 'build_existing_model.geometry_building_type_recs'
        restrict_heating_fuel = 'build_existing_model.heating_fuel'

        # For "resstock_amy2012", convert upgrade 5 to proper aws upgrade, 3
        if run_type == 'resstock_amy2012' and aws_upgrade == '5':
            aws_upgrade = '3'

        # Make aws_cols compatible with the AWS query in non-OEDI cases
        aws_cols.remove('county_name')

        # Create new `counties` list in `<state>, <county>` format for BSQ
        ## Only include df rows where 'in.county' is in the 'counties' list
        df_filtered = df_meta[df_meta['in.county'].isin(counties)]
        
        ## 'in.state' and 'in.county_name' in the format '<state>, <county>'
        df_filtered['state_county'] = (df_filtered['in.state'] + ', ' 
                                       + df_filtered['in.county_name'])
        
        ## Convert 'state_county' column to a list and remove duplicates
        aws_counties = list(set(df_filtered['state_county'].tolist()))

    # Create a BuildStockQuery object based on the run type
    my_run = BuildStockQuery(**aws_run_type)

    # TODO? Simplify my_run.X.Y() with dict unpacking (**) to avoid duplication

    # AWS QUERY/BUILDSTOCKQUERY
    if sw_savings_shape and is_oedi:
        ts_savings_query = my_run.savings.savings_shape(
            upgrade_id=0 if sw_comstock else aws_upgrade,
            enduses=elec_enduse + natural_gas,
            restrict=[('applicability', [True]),
                      (restrict_county, aws_counties),
                      (restrict_heating_fuel, heating_fuel)],
            group_by=aws_cols,
            get_query_only=True,
            annual_only=False
        )

        chunk_states_str = ', '.join(f"'{state}'" for state in chunk_states)

        ts_savings_query = (
            ts_savings_query.replace(
                'SELECT ts_b.timestamp',
                'SELECT from_unixtime_nanos(ts_b.timestamp)'
            )
            .replace(
                'upgrade = \'0\') AS ts_u',
                f'upgrade = \'{aws_upgrade}\') AS ts_u'
            )
            .replace(
                'WHERE comstock_amy2018_release_1_by_state.upgrade',
                (
                    'WHERE comstock_amy2018_release_1_by_state.state IN '
                    f'({chunk_states_str}) '
                    'AND comstock_amy2018_release_1_by_state.upgrade'
                )
            )
            .replace(
                'WHERE comstock_amy2018_release_1_metadata.applicability = true',
                (
                    'WHERE comstock_amy2018_release_1_metadata.applicability = true '
                    'AND comstock_amy2018_release_1_metadata.upgrade = '
                    f'{int(aws_upgrade)}'
                )
            )
        )

        print(ts_savings_query)

        # Execute the query and store the results in ts_agg as a DataFrame
        ts_agg = query_execution(ts_savings_query, my_run)

    elif sw_savings_shape and not is_oedi:
        ts_agg = my_run.savings.savings_shape(
            upgrade_id=aws_upgrade,
            enduses=elec_enduse + natural_gas,
            restrict=[('state', chunk_states), # partition in AWS query
                      (restrict_county, aws_counties), # partition in AWS query
                      (restrict_heating_fuel, heating_fuel)],
            group_by=aws_cols,
            timestamp_grouping_func='hour',
            get_query_only=False,
            applied_only=True,
            annual_only=False
        )

    elif is_oedi: # elif not sw_savings_shape and is_oedi
        ts_agg_query = my_run.agg.aggregate_timeseries(
            upgrade_id=0 if sw_comstock else aws_upgrade,
            enduses=elec_enduse + natural_gas,
            restrict=[('state', chunk_states), # partition in the AWS query
                      ('upgrade', [int(aws_upgrade)]), # partition in the AWS query
                      *([('applicability', [True])] if applied_only else []),
                      (restrict_county, aws_counties),
                      (restrict_heating_fuel, heating_fuel)],
            group_by=aws_cols,
            get_query_only=True,
            annual_only=False
        )

        # ts_agg is a query string; Modify for OEDI data query
        table_name = aws_run_type['table_name']
        ts_agg_query = ts_agg_query.replace(
            f'{table_name}_by_state.timestamp',
            f'from_unixtime_nanos({table_name}_by_state.timestamp)')

        if sw_comstock:
            # Comstock OEDI query string requires additional modification
            ts_agg_query = ts_agg_query.replace(
              f"metadata.upgrade = {aws_upgrade}",
              f"metadata.upgrade = {aws_upgrade} AND "
              f"comstock_amy2018_release_1_by_state.upgrade = '{aws_upgrade}'"
            )
            if comstock_year == "2024" and comstock_release == "2":
                ts_agg_query = (
                    ts_agg_query.replace(
                        'comstock_amy2018_release_1_by_state',
                        'comstock_2024_amy2018_release_2_by_state'
                    )
                    .replace(
                        'comstock_2024_amy2018_release_1_by_state',
                        'comstock_2024_amy2018_release_2_by_state'
                    )
                    .replace(
                        'comstock_2024_amy2018_release_1_metadata',
                        'comstock_amy2018_r2_2024_md_agg_by_state_and_county_parquet'
                    )
                    .replace(
                        f"_by_state.upgrade = '{aws_upgrade}'",
                        f"_by_state.upgrade = {aws_upgrade}"
                    )
                    .replace(
                        'in.county_name',
                        'county'
                    )
                )

        print(ts_agg_query)

        # Execute the query and store the results in ts_agg as a DataFrame
        ts_agg = query_execution(ts_agg_query, my_run)

    else: # elif not sw_savings_shape and not is_oedi
        # TODO(?): Incorporate 'state' and 'upgrade' into restrict list
        ts_agg = my_run.agg.aggregate_timeseries(
            upgrade_id=aws_upgrade,
            enduses=elec_enduse + natural_gas,
            restrict=[('state', chunk_states), # partition in the AWS query
                      (restrict_county, aws_counties),
                      (restrict_heating_fuel, heating_fuel)],
            group_by=aws_cols,
            timestamp_grouping_func='hour',
            get_query_only=False,
            applied_only=applied_only,
            annual_only=False
        )

    # POST-PROCESSING THE DATA RECEIVED FROM AWS QUERY/BUILDSTOCKQUERY
    # Trim '__baseline' from column names if sw_savings_shape
    if sw_savings_shape:
        ts_agg.columns = [col.replace('__baseline', '')
                          for col in ts_agg.columns]
    
    if is_oedi:
        # Remove 'out.' from elec_enduse column names to correspond to ts_agg
        elec_enduse = [item.replace('out.', '') for item in elec_enduse]

    else:
        # Reinsert 'county_name' into aws_cols
        aws_cols = bsq_cols.copy()

        # Make ts_agg['county] just the county name
        ts_agg['county_name'] = ts_agg['county'].str.split(', ').str[1]

        # Replace 'county' in ts_agg with matching 'in.county' in df_filtered
        cnty_to_nhgis_cnty = (
            df_filtered.set_index('state_county')['in.county']
                       .to_dict())
        ts_agg['county'] = ts_agg['county'].map(cnty_to_nhgis_cnty)

        ts_agg.rename(columns={'time': 'timestamp'}, inplace=True)

        # Set timestamp to hour end not hour start
        ts_agg['timestamp'] = ts_agg['timestamp'] + pd.Timedelta(hours=1)

        # Sum the natural gas columns and convert from kbtu to kWh
        ts_agg['natural_gas.heating.energy_consumption'] = (
            (ts_agg[natural_gas].sum(axis=1)) * 0.293071
        )

    if sw_comstock:
        # Set the timestamp as the index
        ts_agg.set_index('timestamp', inplace=True)

        # Shift index by 59 minutes for resampling (timestamp will be hour end)
        ts_agg.index = ts_agg.index + pd.Timedelta(minutes=59)
        
        # Sum each hour's energy consumption for each unique bsq_cols group
        ts_agg = ts_agg.groupby(bsq_cols).resample('H').sum()

        ts_agg = ts_agg.drop(columns=bsq_cols)

        ts_agg.reset_index(inplace=True)

        if comstock_year == "2024" and comstock_release == "2":
            state_county_map = pd.read_csv(
                os.path.join(
                    output_dir, "inputs", "spatial_tract_lookup_table.csv")
            )

            # Merge state_county_map w/ df_meta to bring in resstock_county_id
            ts_agg = ts_agg.merge(
                state_county_map[
                    ["nhgis_county_gisjoin", "resstock_county_id"]
                ].drop_duplicates(),
                how="left",
                on="nhgis_county_gisjoin"
            )

            # Assign resstock_county_id to in.county_name
            ts_agg["county_name"] = ts_agg["resstock_county_id"]

    # Add building ID column from groupby columns and set as index
    ts_agg['bldg_id'] = ts_agg[aws_cols].apply(tuple, axis=1).astype(str)
    ts_agg.set_index('bldg_id', inplace=True)

    # Sum energy consumption of cooling and heating as HVAC.elec
    ts_agg['HVAC.elec'] = ts_agg[elec_enduse].sum(axis=1)

    ts_agg = ts_agg[['timestamp', 'HVAC.elec',
                     'natural_gas.heating.energy_consumption']]
    
    # Convert HVAC.elec and natural_gas columns from kWh to MWh & round
    ts_agg['HVAC.elec'] = (ts_agg['HVAC.elec'] / 1000).round(6)
    ts_agg['natural_gas.heating.energy_consumption'] = (
        ts_agg['natural_gas.heating.energy_consumption'] / 1000).round(6)

    return ts_agg

def weather_data(url_base, year, state, county_id, max_retries=60, delay=60):
    """
    Retrieves weather data from a URL and performs data preprocessing.

    Parameters:
    url_base (str): The base URL for the weather data.
    year (int): The year for which the weather data is retrieved.
    state (str): The state for which the weather data is retrieved.
    county_id (str): The county ID for which the weather data is retrieved.
    max_retries (int): The maximum number of retries for a server error.
    delay (int): The delay in seconds between retries.

    Returns:
    df_weather (DataFrame): The preprocessed weather data as a DataFrame.
    """
    print(f'Retrieving weather data for {year} in {state} county {county_id}.')
    # Read the weather data into a DataFrame
    url_weather = (url_base + 
        f'2022/resstock_amy{year}_release_1/weather/'
        f'state={state}/{county_id}_{year}.csv')

    for _ in range(max_retries):
        try:
            df_weather = pd.read_csv(url_weather,
                                     parse_dates=True,
                                     index_col='date_time')
            break
        except HTTPError as e:
            if e.code in [500, 503]:
                print(f"Server error {e.code}, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise
    else:
        traceback.print_exc()
        raise Exception("Server error after multiple retries")

    df_weather.index = pd.to_datetime(df_weather.index)

    # Adding lagged features for the dry bulb temperature
    for lag in lag_hours:
        df_weather[f'Dry Bulb Temperature Lag {lag}h'] = df_weather['Dry Bulb Temperature [°C]'].shift(lag)

    # Fill NaN values caused by the lagging operation
    df_weather.fillna(method='bfill', inplace=True)
    df_weather.fillna(method='ffill', inplace=True)

    # Add a column for the time of day as a float
    df_weather['Time of Day'] = df_weather.index.hour
    # Add a column for weekend or weekday as a binary value
    df_weather['Weekend'] = df_weather.index.weekday.isin([5, 6]).astype(int)

    # Reset the index for further operations
    df_weather.reset_index(drop=True, inplace=True)

    # Select and return the final set of columns including the new lagged features
    columns_to_keep = ['Dry Bulb Temperature [°C]',
        'Relative Humidity [%]', 'Wind Speed [m/s]', 'Wind Direction [Deg]',
        'Global Horizontal Radiation [W/m2]', 'Direct Normal Radiation [W/m2]',
        'Diffuse Horizontal Radiation [W/m2]', 'Time of Day', 'Weekend']
    # Append dynamically generated lag column names
    columns_to_keep.extend([f'Dry Bulb Temperature Lag {lag}h' for lag in lag_hours])
    df_weather = df_weather[columns_to_keep]

    return df_weather

def test_fit(yr_type, year, prefix, upgrade, bldg_id, model, Y_test, Y_pred, 
             X_train, sw_save_metrics, output_dir, sw_save_fit, sw_show_fit, i,
             df_meta, Y, start_index, end_index, energy_type):
    """
    Perform testing and evaluation of the random forest model.

    Args:
        yr_type (str): The type of year for which the model is being tested.
        year (int): The year for which the model is being tested.
        prefix (str): Prefix string for identification.
        upgrade (int): Upgrade identifier.
        bldg_id (int): Building identifier.
        model (RandomForestRegressor): Random forest model.
        Y_test (pd.Series): Test data target.
        Y_pred (np.array): Predicted target values.
        X_train (pd.DataFrame): Training data features.
        sw_save_metrics (bool): Flag to save metrics to a file.
        output_dir (str): The output directory path.
        sw_save_fit (bool): Flag to save the fit plot to a file.
        sw_show_fit (bool): Flag to show the fit plot.
        i (int): Index of the loop.
        df_meta (pd.DataFrame): Metadata DataFrame.
        Y (pd.Series): Actual values.
        start_index (int): The starting index of the chunk.
        end_index (int): The ending index of the chunk.
        energy_type (str): 'HVAC.elec' or 'natural_gas.heating.energy_consumption'

    Returns:
        None
    """

    """ DELETE: TODO: Should round, etc. _after_ calculations for average
        metrics. May want to create global df instead of appending .csv. For
        now, it's good enough. Also, transpose columns in averages.csv?"""

    energy_out = 'HVAC.elec' if energy_type == 'HVAC.elec' else 'natural_gas'
    # Print the building ID/upgrade/year combination
    print(f'{yr_type}{year}_{prefix}up{upgrade:02}_{str(bldg_id)}_{energy_out}')
    # Calculate the metrics
    mae = format(mean_absolute_error(Y_test, Y_pred), '.3g')
    mse = format(mean_squared_error(Y_test, Y_pred), '.3g')
    r2 = round(r2_score(Y_test, Y_pred), 3)
    feature_importances = [round(importance, 3) 
                           for importance in model.feature_importances_]

    # Print metrics to terminal
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R2 Score: {r2}')
    print('Feature Importances:')
    for feature, importance in zip(X_train.columns, feature_importances):
        print(f'    {feature}: {importance}')
    print('\n')

    fig_dir = f'{output_dir}/{yr_type}{year}'
    metrics = f'{prefix}metrics_upgrade{upgrade}_{start_index:04}-{end_index:04}'
    if sw_save_metrics:
        os.makedirs(fig_dir, exist_ok=True) # DELETE TODO: duplicated code that will unnecessarily run for each building
        # Save the metrics and feature importances to a .txt file
        with open(f'{fig_dir}/{metrics}.txt', 'a') as f:
            f.write(f'{prefix}up{upgrade:02}_{str(bldg_id)}_{energy_out}\n')
            f.write(f'MAE: {mae}\n')
            f.write(f'MSE: {mse}\n')
            f.write(f'R2 Score: {r2}\n')
            f.write('Feature Importances:\n')
            for feature, importance in zip(X_train.columns,
                                           feature_importances):
                f.write(f'    {feature}: {importance}\n')
            f.write('\n')

        ## Output metrics to a CSV file
        # Create a dictionary with metrics and feature importances
        data = {'Building ID': [
                    f'{prefix}up{upgrade:02}_{str(bldg_id)}_{energy_out}'],
                'MAE': [mae],
                'MSE': [mse],
                'R2 Score': [r2]}
        for feature, importance in zip(X_train.columns, feature_importances):
            data[f'FI - {feature}'] = [importance]
        # Convert the dictionary to a DataFrame and round the values
        df = pd.DataFrame(data)
        # Check if the file exists
        file_exists = os.path.isfile(f'{fig_dir}/{metrics}.csv')
        # Append the DataFrame to a CSV file
        df.to_csv(f'{fig_dir}/{metrics}.csv',
                  mode='a', index=False, header=not file_exists)

        ## Output average metrics to a CSV file
        # Check if it's the last iteration of the loop
        if i == len(df_meta.index) - 1:
            # Read the metrics file into a DataFrame
            df = pd.read_csv(f'{fig_dir}/{metrics}.csv')
            # Calculate the average of each column
            averages = df.drop(columns='Building ID').mean().round(6)
            # Convert the Series to a DataFrame and transpose it
            averages_df = pd.DataFrame(averages)
            # Write the DataFrame to a CSV file
            averages_df.to_csv(f'{fig_dir}/averages_{prefix}metrics{upgrade}_{start_index:04}-{end_index:04}.csv', header=False)

    # Output and/or show the fit plot
    if sw_save_fit or sw_show_fit:
        if yr_type == 'targ':
            # Create a new figure with a specified size
            plt.figure(figsize=(100, 10))  # width and height in inches
            # Plot Y and Y_pred
            plt.plot(Y, label='Actual')
            plt.plot(Y_pred, label='Predicted')
            # Set the title and labels
            plt.title('HVAC.elec over time')
            plt.xlabel('Time')
            plt.ylabel('HVAC.elec')
            # Show the legend
            plt.legend()
            if sw_save_fit:
                os.makedirs(fig_dir, exist_ok=True) # DELETE TODO: duplicated code that will unnecessarily run for each building
                plt.savefig(f'{fig_dir}/{prefix}up{upgrade:02}_{str(bldg_id)}.png')
            if sw_show_fit:
                plt.show()

        # Create a scatter plot of the actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_test, Y_pred, color='blue')
        plt.plot([min(Y_test), max(Y_test)], 
                 [min(Y_test), max(Y_test)], color='red')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        if sw_save_fit:
            os.makedirs(fig_dir, exist_ok=True) # DELETE TODO: duplicated code that will unnecessarily run for each building
            plt.savefig(f'{fig_dir}/fit_{prefix}up{upgrade:02}_{str(bldg_id)}.png')
        if sw_show_fit:
            plt.show()

def prediction(base_year, df_eulp, sw_test_base, target_year, sw_test_target,
               energy_type, weather_data_base, weather_data_target, bldg_id,
               df_eulp_targ):
    """
    Predicts hourly HVAC electricity consumption for a single building in a 
    target year given known HVAC electricity consumption in a base year and
    known weather for the state/county in the base year and target year.

    Args:
        state (str): The state for which to make the prediction.
        county_id (int): The ID of the county for which to make the prediction.
        base_year (int): The base year for training the prediction model.
        df_eulp (DataFrame): df containing the electricity consumption data.
        sw_test_base (bool): Whether to test/evaluate the random forest model.
        url_base (str): The base URL for weather data.
        target_year (int): The target year for which to make the prediction.
        sw_test_target (bool): Whether to test/evaluate the random forest model.
        energy_type (str): 'HVAC.elec' or 'natural_gas.heating.energy_consumption'
        weather_data_base (DataFrame): The weather data for the base year.
        weather_data_target (DataFrame): The weather data for the target year.
        bldg_id (str): The building ID for which to make the prediction.
        df_eulp_targ (DataFrame): Electricity consumption data for target year.

    Returns:
        predictions (DataFrame): df containing the predicted HVAC electricity 
            consumption for each timestamp in the target year.
    """

    # Drop the 1st entry (1:00a) of the EULP data bc the weather data starts at
    # 1:00a (instantaneous) and the EULP data for 2:00a is 1:00a to 2:00a
    df_eulp = df_eulp.iloc[1:]
    Y = df_eulp[energy_type].reset_index(drop=True)

    # Return target year EULP of zeros if sum of base year EULP is near zero
    if Y.sum() <= 0.01:
        index_array = range(len(weather_data_target))
        predictions = pd.DataFrame(
            np.zeros(len(weather_data_target)), index=index_array
        )
        predictions['timestamp'] = pd.date_range(
            start=dt.datetime(target_year, 1, 1, 1, 0, 0),
            end=dt.datetime(target_year+1, 1, 1, 0, 0, 0),
            freq='H'
        )
        predictions = predictions.rename(columns={0: energy_type})
        predictions = predictions[['timestamp', energy_type]]
        print(f'{energy_type} is all zeros for {target_year} for {bldg_id}.')
        return predictions
    # Pull in the weather data for the base year
    X = weather_data_base
    # Drop final row so the lengths of the weather and EULP data match
    X = X.iloc[:-1]
    startTime = dt.datetime.now()

    # Train random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    if sw_cross_val:
        # Perform 5-fold cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        results = cross_val_score(rf_model, X, Y, cv=kfold, scoring='neg_mean_squared_error')
        print(f'Cross-validated MSE: {results.mean()}')
        rf_model.fit(X, Y)
        print('Finished Random Forest Model Training: '+ str(dt.datetime.now() - startTime))
        if sw_test_base or sw_save_metrics:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            Y_pred = rf_model.predict(X_test)
            test_fit('base', base_year, prefix, upgrade, bldg_id, rf_model, Y_test,
                    Y_pred, X_train, sw_save_metrics, output_dir, sw_save_fit,
                    sw_show_fit, i, df_meta, Y, start_index, end_index, 
                    energy_type)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                            random_state=42)  
        rf_model.fit(X_train, Y_train)
        print('Finished Random Forest Model Training: '+ str(dt.datetime.now() - startTime))
        if sw_test_base or sw_save_metrics:
            # Make predictions on the test data
            Y_pred = rf_model.predict(X_test)
            test_fit('base', base_year, prefix, upgrade, bldg_id, rf_model, 
                     Y_test, Y_pred, X_train, sw_save_metrics, output_dir,
                     sw_save_fit, sw_show_fit, i, df_meta, Y, start_index,
                     end_index, energy_type)
    X_Predict = weather_data_target
    rf_predictions = rf_model.predict(X_Predict)

    # Determine if predictions require extrapolation using the Neural Network
    min_train = X['Dry Bulb Temperature [°C]'].min()
    max_train = X['Dry Bulb Temperature [°C]'].max()
    min_predict = X_Predict['Dry Bulb Temperature [°C]'].min()
    max_predict = X_Predict['Dry Bulb Temperature [°C]'].max()
    needs_extrapolation = min_predict < min_train or max_predict > max_train

    # If hybrid model is on and extrapolation needed, use RFR + NN (v just RFR)
    if sw_hybrid_model and needs_extrapolation:
        print('Extrapolation required, employing hybrid model.')
        scaler = StandardScaler()
        X_scale = scaler.fit_transform(X)

        # Define a simple neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                128,
                activation='relu',
                input_shape=(X_scale.shape[1],),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
            ),
            tf.keras.layers.Dense(
                64,
                activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
            ),
            tf.keras.layers.Dense(
                1,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)
            )
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        startTime = dt.datetime.now()
        count_rf = 0
        count_nn = 0

        if sw_cross_val:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_index, test_index in kfold.split(X_scale):
                X_train, X_test = X_scale[train_index], X_scale[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                model.fit(X_train, Y_train, epochs=50, batch_size=10, verbose=0)
                mse = model.evaluate(X_test, Y_test, verbose=0)
                print(f'Fold MSE: {mse}')
            print('Finished Neural Network Training with Cross Validation: ' + str(dt.datetime.now() - startTime))
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.2, random_state=42)
            model.fit(X_train, Y_train, epochs=50, batch_size=10)
            print('Finished Neural Network Training without Cross Validation: ' + str(dt.datetime.now() - startTime))

        X_Predict_scale = scaler.transform(X_Predict)
        nn_predictions = model.predict(X_Predict_scale).flatten()

        predictions = []
        for j in range(len(X_Predict)):
            if ((X_Predict['Dry Bulb Temperature [°C]'].iloc[j] >= min_train) & (X_Predict['Dry Bulb Temperature [°C]'].iloc[j] <= max_train)):
                predictions.append(rf_predictions[j])
                count_rf += 1
            else:
                predictions.append(nn_predictions[j])
                count_nn += 1
        print('Total extrapolation percentage: ' + str((count_nn / (count_rf + count_nn)) * 100) + '%')

    else:
        predictions = rf_predictions
        print('Using random forest predictions as no extrapolation is needed.')

    index_array = range(len(predictions))
    predictions = pd.DataFrame(predictions, index=index_array)
    # 'timestamp' represents hour-end values, so, like above, a 1:00 AM entry
    # for weather data corresponds to the 2:00 AM (1a-2a) entry for EULP data
    start_datetime = dt.datetime(target_year, 1, 1, 2, 0, 0)
    end_datetime = dt.datetime(target_year+1, 1, 1, 1, 0, 0)
    datetime_range_predictions = pd.date_range(start=start_datetime,
                                               end=end_datetime, freq='H')
    predictions['timestamp'] = datetime_range_predictions
    predictions = predictions.rename(columns={0: energy_type})
    predictions = predictions[['timestamp', energy_type]]
    ## Adjust the predictions DataFrame to match the EULP data
    ## with timestamps representing hour-end values
    # Create a new DataFrame with the first 1:00 AM entry
    first_entry = pd.DataFrame({
        'timestamp': [pd.to_datetime(predictions['timestamp'].iloc[0].date())
                      + pd.Timedelta(hours=1)],
        energy_type: [np.nan]})
    # Append the original DataFrame to the new DataFrame
    predictions = pd.concat([first_entry, predictions], ignore_index=True)
    # Use the energy_type value for 2:00 AM to backfill for 1:00 AM
    predictions.loc[0, energy_type] = predictions.loc[1, energy_type]
    # Drop the last entry of the DataFrame
    predictions = predictions[:-1]

    # Make sure no negative values are present in the predictions & clip to 0
    predictions[energy_type] = predictions[energy_type].clip(lower=0)

    if sw_test_target:
        # Subset the target year EULP data to the building ID
        df_eulp_targ_bldg = df_eulp_targ.loc[bldg_id].copy()

        Y_test = df_eulp_targ_bldg[energy_type]
        Y = df_eulp_targ_bldg[energy_type].reset_index(drop=True)
        Y_pred = predictions[energy_type]

        test_fit('targ', target_year, prefix, upgrade, bldg_id, rf_model, Y_test,
                 Y_pred, X_Predict, sw_save_metrics, output_dir, sw_save_fit,
                 sw_show_fit, i, df_meta, Y, start_index, end_index,
                 energy_type)
    return predictions


# MAIN
# Load the metadata DataFrame
df_meta = pd.read_csv(meta_path)

# Set `county` based on `sw_comstock` value
county = 'in.nhgis_county_gisjoin' if sw_comstock else 'in.county'

# Subset df_meta to the specified range of counties
counties = counties_str.split('_')
df_meta = df_meta[df_meta[county].isin(counties)]

# Get the unique states in the metadata DataFrame for process_chunk_agg fxn
chunk_states = df_meta['in.state'].unique().tolist()

# Set index of df_meta to 'bldg_id'
df_meta = df_meta.set_index('bldg_id')

# Call function to get aggregate timeseries data
ts_agg = process_chunk_agg(
    base_run, upgrade, counties, bsq_cols, sw_comstock, chunk_states,
    sw_savings_shape, df_meta, applied_only
)

# Grab the target year AWS data if sw_test_target else set as None
df_eulp_targ = (
    process_chunk_agg(
        target_run, upgrade, counties, bsq_cols, sw_comstock, chunk_states,
        sw_savings_shape, df_meta, applied_only
    )
    if sw_test_target and sw_apply_regression
    else None
)

# Error check: Sum AWS HVAC and ng timeseries data for each bldg_id in df_meta
df_meta['AWS_HVAC.elec'] = ts_agg.groupby('bldg_id').apply(lambda x: (
    x['HVAC.elec'].iloc[:8760].sum()))

df_meta['AWS_natural_gas.heating.energy_consumption'] = (
    ts_agg.groupby('bldg_id').apply(lambda x: (
    x['natural_gas.heating.energy_consumption'].iloc[:8760].sum())))

# Create an empty list to store DataFrames for each building
df_bldg = []

if sw_apply_regression: # TODO: or `individual_building`?
    # Create a dictionary to store weather data for each unique county
    weather_data_dict = {}

    # Loop through each building in the metadata DataFrame
    for i, bldg_id in enumerate(df_meta.index):
        # Get the state of the building (for the URL)
        state = df_meta.loc[bldg_id, 'in.state']
        # Get the county ID of the building
        county_id = df_meta.loc[bldg_id, county]

        # Check if weather data for this county is already in the dictionary
        if county_id not in weather_data_dict:
            # If not, get & store weather data for base & target years in dict
            weather_data_dict[county_id] = {
                'base': weather_data(url_base, base_year, state, county_id),
                'target': weather_data(url_base, target_year, state, county_id)
            }

        # Get the EULP data for a specific building for use in the regressions
        df_eulp_pred = ts_agg.loc[bldg_id].copy()

        # HVAC ELECTRICITY
        # Predict HVAC electricity energy consumption
        df_eulp_hvac = prediction(
            base_year, df_eulp_pred, sw_test_base, target_year, sw_test_target,
            'HVAC.elec', weather_data_dict[county_id]['base'],
            weather_data_dict[county_id]['target'], bldg_id, df_eulp_targ
        )

        # Error check: Add regressed annual HVAC.elec to df_meta for a bldg_id
        df_meta.loc[bldg_id, 'HVAC.elec'] = (
            df_eulp_hvac['HVAC.elec'].iloc[:8760].sum().round(6))

        # Rename columns, including `HVAC.elec` to match the building ID
        df_eulp_hvac.columns = ['timestamp_EST', f'{bldg_id}']

        # Set the timestamp column to the index
        df_eulp_hvac.set_index('timestamp_EST', inplace=True)

        # Append single building DataFrame to list of building DataFrames
        df_bldg.append(df_eulp_hvac)

        # NATURAL GAS
        # Predict natural gas heating energy consumption
        df_eulp_ng = prediction(
            base_year, df_eulp_pred, sw_test_base, target_year, sw_test_target,
            'natural_gas.heating.energy_consumption',
            weather_data_dict[county_id]['base'],
            weather_data_dict[county_id]['target'], bldg_id, df_eulp_targ
        )

        # Add regressed annual ng to df_meta for a bldg_id
        df_meta.loc[bldg_id, 'natural_gas.heating.energy_consumption'] = (
            df_eulp_ng['natural_gas.heating.energy_consumption']
            .iloc[:8760].sum().round(6))

    # Concatenate all building DataFrames into a single DataFrame
    df_eulp = pd.concat(df_bldg, axis=1)

else:
    # If not applying regression, duplicate annual ng and HVAC columns
    df_meta['HVAC.elec'] = df_meta['AWS_HVAC.elec']
    df_meta['natural_gas.heating.energy_consumption'] = (
        df_meta['AWS_natural_gas.heating.energy_consumption'])

    # Filter ts_agg to include only rows with bldg_id's in df_meta.index
    ts_agg = ts_agg[ts_agg.index.isin(df_meta.index)]

    # Create a timeseries x bldg_id DataFrame
    ts_agg = ts_agg.reset_index()
    ts_agg.rename(columns={'timestamp': 'timestamp_EST'}, inplace=True)
    df_eulp = ts_agg.pivot(index='timestamp_EST', columns='bldg_id',
                           values='HVAC.elec')

# Error checking using ratios and percent differences in df_meta TODO: fxn?
## Ratios (note: small_number is to avoid division by zero)
df_meta['ratio_HVAC_AWS_meta'] = (
    df_meta['AWS_HVAC.elec'] /
    df_meta['meta_HVAC.elec']).round(4)
df_meta['ratio_ng_AWS_meta'] = (
    df_meta['AWS_natural_gas.heating.energy_consumption'] /
    df_meta['meta_natural_gas.heating.energy_consumption']).round(4)

df_meta['ratio_HVAC_reg_meta'] = (
    df_meta['HVAC.elec'] /
    df_meta['meta_HVAC.elec']).round(4)
df_meta['ratio_ng_reg_meta'] = (
    df_meta['natural_gas.heating.energy_consumption'] /
    df_meta['meta_natural_gas.heating.energy_consumption']).round(4)

df_meta['ratio_HVAC_reg_AWS'] = (
    df_meta['HVAC.elec'] /
    df_meta['AWS_HVAC.elec']).round(4)
df_meta['ratio_ng_reg_AWS'] = (
    df_meta['natural_gas.heating.energy_consumption'] /
    df_meta['AWS_natural_gas.heating.energy_consumption']).round(4)

## Percent differences (note: small_number is to avoid division by zero)
df_meta['diff_HVAC_AWS_meta'] = (100 * (
    (df_meta['AWS_HVAC.elec'] - df_meta['meta_HVAC.elec']) /
    df_meta['meta_HVAC.elec'])).round(4)
df_meta['diff_ng_AWS_meta'] = (100 * (
    (df_meta['AWS_natural_gas.heating.energy_consumption'] -
     df_meta['meta_natural_gas.heating.energy_consumption']) /
    df_meta['meta_natural_gas.heating.energy_consumption'])).round(4)

df_meta['diff_HVAC_reg_meta'] = (100 * (
    (df_meta['HVAC.elec'] - df_meta['meta_HVAC.elec']) /
    df_meta['meta_HVAC.elec'])).round(4)
df_meta['diff_ng_reg_meta'] = (100 * (
    (df_meta['natural_gas.heating.energy_consumption'] -
     df_meta['meta_natural_gas.heating.energy_consumption']) /
    df_meta['meta_natural_gas.heating.energy_consumption'])).round(4)

df_meta['diff_HVAC_reg_AWS'] = (100 * (
    (df_meta['HVAC.elec'] - df_meta['AWS_HVAC.elec']) /
    df_meta['AWS_HVAC.elec'])).round(4)
df_meta['diff_ng_reg_AWS'] = (100 * (
    (df_meta['natural_gas.heating.energy_consumption'] -
     df_meta['AWS_natural_gas.heating.energy_consumption']) /
    df_meta['AWS_natural_gas.heating.energy_consumption'])).round(4)

df_meta.rename(columns={
    "in.geometry_building_type_recs": "in.building_type",
    "in.comstock_building_type": "in.building_type",
    "in.county": "in.county_nhgis",
    "in.nhgis_county_gisjoin": "in.county_nhgis",
}, errors='ignore', inplace=True)

# Create 'gas_heating_MWh' column and insert after bsq_cols & sqft columns
df_meta.insert(len(bsq_cols) + 1, 'gas_heating_MWh',
               df_meta['natural_gas.heating.energy_consumption'])

# Save metadata DataFrame to CSV file
df_meta.to_csv(os.path.join(output_dir,
    f'{prefix}meta_upgrade{upgrade}_{start_index:04}-{end_index:04}.csv'))

# Round df_eulp to 6 decimal places TODO: Move to `prediction` fxn?
df_eulp = df_eulp.round(6)

# If leap year (currently resstock), drop the last day of `df_eulp`
if len(df_eulp) == 8784:
    df_eulp = df_eulp.iloc[:-24]

# Save concatenated energy consumption DataFrame (hour x bldg) to CSV file
df_eulp.to_csv(os.path.join(output_dir,
    f'{prefix}eulp_hvac_elec_MWh_upgrade{upgrade}_'
    f'{start_index:04}-{end_index:04}.csv'))

print('\nChunk done at:', dt.datetime.now())
print('Total time elapsed:', dt.datetime.now() - script_start_time)