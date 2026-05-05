# IMPORTS
# Import libraries
import os
import certifi
_CA = certifi.where()
os.environ.setdefault("AWS_CA_BUNDLE", _CA)
os.environ.setdefault("CURL_CA_BUNDLE", _CA)
os.environ.setdefault("SSL_CERT_FILE", _CA)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
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
import re
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set environment variable to disable OneDNN prior to importing tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Defer TensorFlow import to avoid forking a TF-initialized runtime.
# Import and seed only when/if the hybrid model is used.
_TF = None
def _ensure_tf():
    global _TF
    if _TF is None:
        import tensorflow as tf
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        _TF = tf
    return _TF

def _parse_year_entry(entry):
    """
    Parse one target year entry.
    Supported forms:
      - 2018
      - "2018"
      - "2007-2013"
    """
    if isinstance(entry, int):
        return [entry]
    if isinstance(entry, str):
        token = entry.strip()
        if not token:
            return []
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid year range: {entry}")
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if end < start:
                raise ValueError(f"Year range end < start: {entry}")
            return list(range(start, end + 1))
        return [int(token)]
    raise TypeError(f"Unsupported target_year entry type: {type(entry)}")

def parse_target_years(year_spec):
    """
    Parse target_year setting into a sorted unique list of years.
    Supported forms:
      - 2018
      - "2018"
      - ["2007-2013", 2016, "2018"]
    """
    if isinstance(year_spec, (int, str)):
        years = _parse_year_entry(year_spec)
    elif isinstance(year_spec, list):
        years = []
        for entry in year_spec:
            years.extend(_parse_year_entry(entry))
    else:
        raise TypeError(f"Unsupported target_year type: {type(year_spec)}")

    years = sorted(set(years))
    if not years:
        raise ValueError("target_year parsed to an empty year list")
    return years

script_start_time = dt.datetime.now()
print('Script start time:', script_start_time)

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
sleep_seconds = switch['sleep_seconds'] # Number of seconds to sleep at the start of the script to prevent AWS token errors when multiple jobs are run simultaneously
## Columns to group by; Note: if this changes, language in the 
## `process_chunk_agg` function will need to be updated following the AWS call
bsq_cols =  switch['com_bsq_cols'] if sw_comstock else switch['res_bsq_cols']
## Note: resstock upgrades do not correspond to the same # as comstock
upgrades = switch['upgrades'] # default: comstock = [0, 1, 14, 55], resstock = [0, 4, 8]
## Number of buildings to pull per upgrade
n_bldngs = switch['n_bldngs'] # 'all' for all buildings, 'assign' for assigned building id list from csv
base_year = switch['base_year'] # Base year for the building stock
target_years = parse_target_years(switch['target_year']) # Target weather years
comparison_year = (
    base_year if base_year in target_years else target_years[0]
) # Year used for df_meta annual regression comparison columns
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
weather_data_base = switch['weather_data_base']

if sw_test_target and len(target_years) != 1:
    raise ValueError(
        "switches_agg.json: sw_test_target=True requires exactly one target_year."
    )

# Force program to sleep for a random amount of seconds between 0 and sleep_seconds
# Prevents AWS token errors when multiple jobs are run simultaneously
time.sleep(random.uniform(0, sleep_seconds))

# FUNCTIONS
# Detect HPC (SLURM or explicit flag)
def _is_hpc() -> bool:
    return bool(int(os.environ.get('SLURM_JOB_ID', 0) or
                    os.environ.get('REEDS_USE_SLURM', 0)
                    ))

# Determine county column based on comstock/resstock and version for HPC
def _county_of(bid):
    if sw_comstock and comstock_year == "2025" and comstock_release == "2":
        return df_meta.loc[bid, 'in.as_simulated_nhgis_county_gisjoin']
    else:
        return df_meta.loc[bid, county]

# Process one building worker function - for HPC multiprocessing
def _process_one_building(args):
    """
    Worker: run predictions for one building and return:
      - building HVAC timeseries df (timestamp_EST x 1 column)
      - annual sums for HVAC and NG for df_meta
    Notes:
      - Metrics and plots are disabled in workers to avoid file contention.
    """
    (bldg_id,
     county_id,
     df_eulp_pred,
     df_eulp_targ_local,
     base_year,
     target_years,
     sw_test_base,
     sw_test_target) = args

    # Ensure globals used in prediction/test_fit exist but do nothing noisy
    global i, sw_save_metrics, sw_show_fit, sw_save_fit
    i = 0
    sw_save_metrics = False
    sw_show_fit = False
    sw_save_fit = False

    # Load weather just-in-time for this building.
    weather_base_df = weather_data(weather_data_base, base_year, county_id)
    target_weather_frames = []
    target_year_by_row = []
    for yr in target_years:
        year_df = weather_data(weather_data_base, yr, county_id)
        target_weather_frames.append(year_df)
        target_year_by_row.extend([yr] * len(year_df))
    weather_target_df = pd.concat(target_weather_frames, ignore_index=True)
    target_year_by_row = np.asarray(target_year_by_row)

    # HVAC
    df_eulp_hvac = prediction(
        base_year, df_eulp_pred, sw_test_base, target_years, sw_test_target,
        'HVAC.elec', weather_base_df, weather_target_df, bldg_id, df_eulp_targ_local,
        target_year_by_row
    )
    hvac_sum = (
        df_eulp_hvac.loc[
            (df_eulp_hvac['timestamp'] - pd.Timedelta(hours=1)).dt.year
            == comparison_year,
            'HVAC.elec'
        ].sum().round(6)
    )

    # NG
    df_eulp_ng = prediction(
        base_year, df_eulp_pred, sw_test_base, target_years, sw_test_target,
        'natural_gas.heating.energy_consumption',
        weather_base_df, weather_target_df, bldg_id, df_eulp_targ_local,
        target_year_by_row
    )
    ng_sum = (
        df_eulp_ng.loc[
            (df_eulp_ng['timestamp'] - pd.Timedelta(hours=1)).dt.year
            == comparison_year,
            'natural_gas.heating.energy_consumption'
        ].sum().round(6)
    )

    # Shape HVAC df to timestamp_EST x bldg_id
    df_out = df_eulp_hvac.copy()
    df_out.columns = ['timestamp_EST', f'{bldg_id}']
    df_out.set_index('timestamp_EST', inplace=True)

    return bldg_id, df_out, hvac_sum, ng_sum

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
            print(f"Executing query: {query} \n attempt {attempt + 1}...")
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
    aws_cols = [c for c in bsq_cols]
    aws_counties = counties.copy()
    aws_run_type = run_types[run_type].copy()
    natural_gas = ['out.natural_gas.heating.energy_consumption']

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
        # ResStock has suffix '..kwh' for electricity & ng enduse columns
        elec_enduse = [item + '..kwh' for item in elec_enduse]
        natural_gas = [enduse + '..kwh' for enduse in natural_gas]

    restrict_county = ('in.nhgis_county_gisjoin' if sw_comstock else 
                        'in.county')
    my_run = BuildStockQuery(**aws_run_type)

    ts_agg_query = my_run.query(
        upgrade_id=upgrade,
        applied_only=False,
        enduses=elec_enduse + natural_gas,
        restrict=[('state', chunk_states),
                  (restrict_county, aws_counties)],
        timestamp_grouping_func="hour",
        group_by=aws_cols,
        get_query_only=True,
        annual_only=False
    )

    ts_agg = query_execution(ts_agg_query, my_run)
    elec_enduse = [item.replace('out.', '') for item in elec_enduse]

    # Remove '..kwh' suffix from elec_enduse columns for grouping
    ts_agg.columns = [col.replace('..kwh', '') for col in ts_agg.columns]

    # Remove '..kwh' from elec_enduse list for grouping
    elec_enduse = [item.replace('..kwh', '') for item in elec_enduse]

    if sw_comstock:
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

def weather_data(url_base, year, county_id):
    """
    Retrieves weather data from a URL and performs data preprocessing.

    Parameters:
    url_base (str): The base URL for the weather data.
    year (int): The year for which the weather data is retrieved.
    county_id (str): The county ID for which the weather data is retrieved.

    Returns:
    df_weather (DataFrame): The preprocessed weather data as a DataFrame.
    """
    print(f'Retrieving weather data for {year} county {county_id}.')

    # Local EPW path support (e.g., /projects/geohc/EPW/epw_symlinks)
    epw_path = os.path.join(url_base, f'FIPS_{year}', f'{county_id}_{year}.epw')
    if not os.path.isfile(epw_path):
        raise FileNotFoundError(f'Local EPW file not found: {epw_path}')

    # EPW has 8 metadata lines, then hourly data with no column header.
    df_epw = pd.read_csv(epw_path, skiprows=8, header=None)
    if df_epw.empty:
        raise ValueError(f'Empty EPW file: {epw_path}')

    df_weather = pd.DataFrame({
        'Dry Bulb Temperature [°C]': pd.to_numeric(df_epw.iloc[:, 6], errors='coerce'),
        'Relative Humidity [%]': pd.to_numeric(df_epw.iloc[:, 8], errors='coerce'),
        'Wind Speed [m/s]': pd.to_numeric(df_epw.iloc[:, 21], errors='coerce'),
        'Wind Direction [Deg]': pd.to_numeric(df_epw.iloc[:, 20], errors='coerce'),
        'Global Horizontal Radiation [W/m2]': pd.to_numeric(df_epw.iloc[:, 13], errors='coerce'),
        'Direct Normal Radiation [W/m2]': pd.to_numeric(df_epw.iloc[:, 14], errors='coerce'),
        'Diffuse Horizontal Radiation [W/m2]': pd.to_numeric(df_epw.iloc[:, 15], errors='coerce')
    })

    # Build a time index by row count so downstream features stay consistent.
    df_weather.index = pd.date_range(
        start=f'{year}-01-01 01:00:00',
        periods=len(df_weather),
        freq='h'
    )

    # Add a column for the time of day as a float
    df_weather['Time of Day'] = df_weather.index.hour
    # Add a column for weekend or weekday as a binary value
    df_weather['Weekend'] = df_weather.index.weekday.isin([5, 6]).astype(int)

    # Adding lagged features for the dry bulb temperature
    for lag in lag_hours:
        df_weather[f'Dry Bulb Temperature Lag {lag}h'] = (
            df_weather['Dry Bulb Temperature [°C]'].shift(lag)
        )

    # Fill NaN values caused by the lagging operation
    df_weather.bfill(inplace=True)
    df_weather.ffill(inplace=True)

    # Reset the index for further operations
    df_weather.reset_index(drop=True, inplace=True)

    # Match EULP convention: keep only the first 8760 hourly rows per year.
    df_weather = df_weather.iloc[:8760].copy()

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

def prediction(base_year, df_eulp, sw_test_base, target_years, sw_test_target,
               energy_type, weather_data_base, weather_data_target, bldg_id,
               df_eulp_targ, target_year_by_row):
    """
    Predict hourly energy consumption for a single building by fitting once on
    base-year weather and predicting over concatenated target-year weather.

    Args:
        base_year (int): The base year for training the prediction model.
        df_eulp (DataFrame): df containing the electricity consumption data.
        sw_test_base (bool): Whether to test/evaluate the random forest model.
        target_years (list[int]): Target years for weather-based prediction.
        sw_test_target (bool): Whether to test/evaluate the random forest model.
        energy_type (str): 'HVAC.elec' or 'natural_gas.heating.energy_consumption'
        weather_data_base (DataFrame): The weather data for the base year.
        weather_data_target (DataFrame): Concatenated weather for all target
            years.
        bldg_id (str): The building ID for which to make the prediction.
        df_eulp_targ (DataFrame): Electricity consumption data for target year.
        target_year_by_row (np.ndarray): Per-row target year labels aligned to
            weather_data_target.

    Returns:
        predictions (DataFrame): Predicted hourly consumption across all target
            years, concatenated in target_years order.
    """

    # Drop the 1st entry (1:00a) of the EULP data bc the weather data starts at
    # 1:00a (instantaneous) and the EULP data for 2:00a is 1:00a to 2:00a
    df_eulp = df_eulp.iloc[1:]
    Y = df_eulp[energy_type].reset_index(drop=True)

    # Predict on one concatenated target-weather matrix.
    X_Predict = weather_data_target

    # Return zeros if sum of base year EULP is near zero
    is_zero_base = Y.sum() <= 0.01
    if is_zero_base:
        predictions = np.zeros(len(X_Predict))
        print(f'{energy_type} is all zeros for {target_years} for {bldg_id}.')
        rf_model = None
    else:
        # Pull in the weather data for the base year
        X = weather_data_base
        # Drop final row so the lengths of the weather and EULP data match
        X = X.iloc[:-1]
        startTime = dt.datetime.now()

        # Train random forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
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
            tf = _ensure_tf()
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

    predictions = np.asarray(predictions, dtype=float)

    # Align predictions to hour-end load reporting by year block:
    # value at 02:00 uses weather at 01:00, so shift each year forward by 1 hour.
    shifted = predictions.copy()
    for yr in np.unique(target_year_by_row):
        mask = (target_year_by_row == yr)
        vals = predictions[mask]
        if len(vals) == 0:
            continue
        shifted_vals = vals.copy()
        if len(vals) > 1:
            shifted_vals[1:] = vals[:-1]
        # Fill Jan 1 01:00 with the first modeled value for that year.
        shifted_vals[0] = vals[0]
        shifted[mask] = shifted_vals

    # Build one continuous timestamp vector across all requested target years.
    target_timestamps = []
    for yr in target_years:
        n_rows = int((target_year_by_row == yr).sum())
        target_timestamps.extend(
            pd.date_range(
                start=dt.datetime(yr, 1, 1, 1, 0, 0),
                periods=n_rows,
                freq='H'
            )
        )

    predictions = pd.DataFrame({
        'timestamp': pd.to_datetime(target_timestamps),
        energy_type: shifted
    })
    # Energy consumption cannot be negative.
    predictions[energy_type] = predictions[energy_type].clip(lower=0)

    if is_zero_base:
        return predictions

    if sw_test_target:
        if len(target_years) != 1:
            raise ValueError(
                "sw_test_target=True requires exactly one target year."
            )
        # Subset the target year EULP data to the building ID
        df_eulp_targ_bldg = df_eulp_targ.loc[bldg_id].copy()

        Y_test = df_eulp_targ_bldg[energy_type]
        Y = df_eulp_targ_bldg[energy_type].reset_index(drop=True)
        Y_pred = predictions[energy_type]

        test_fit('targ', target_years[0], prefix, upgrade, bldg_id, rf_model, Y_test,
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
    # Parallel path (HPC only)
    if _is_hpc():
        # Build tasks
        tasks = []
        for bldg_id in df_meta.index:
            county_id = _county_of(bldg_id)
            df_eulp_pred = ts_agg.loc[bldg_id].copy()
            tasks.append((
                bldg_id,
                county_id,
                df_eulp_pred,
                df_eulp_targ if sw_test_target and sw_apply_regression else None,
                base_year,
                target_years,
                sw_test_base,
                sw_test_target
            ))

        # Pool size from CPU count with 4 CPU reserved for cap space
        procs = 48
        print(f'Using {procs} processes for regression out of {os.cpu_count()} possible.')

        df_bldg = []
        with ProcessPoolExecutor(max_workers=procs) as ex:
            futures = [ex.submit(_process_one_building, t) for t in tasks]
            for fut in as_completed(futures):
                bldg_id, df_hvac, hvac_sum, ng_sum = fut.result()
                df_bldg.append(df_hvac)
                df_meta.loc[bldg_id, 'HVAC.elec'] = hvac_sum
                df_meta.loc[bldg_id, 'natural_gas.heating.energy_consumption'] = ng_sum

        df_eulp = pd.concat(df_bldg, axis=1)

    # Serial path (local machine)
    else:
        # Loop through each building in the metadata DataFrame
        for i, bldg_id in enumerate(df_meta.index):
            # Get the county ID of the building
            if sw_comstock and comstock_year == "2025" and comstock_release == "2":
                county_id = df_meta.loc[
                    bldg_id, 'in.as_simulated_nhgis_county_gisjoin'
                ]
            else:
                county_id = df_meta.loc[bldg_id, county]

            # Load weather just prior to regression for this county/building.
            weather_base_df = weather_data(weather_data_base, base_year, county_id)
            target_weather_frames = []
            target_year_by_row = []
            for yr in target_years:
                year_df = weather_data(weather_data_base, yr, county_id)
                target_weather_frames.append(year_df)
                target_year_by_row.extend([yr] * len(year_df))
            weather_target_df = pd.concat(target_weather_frames, ignore_index=True)
            target_year_by_row = np.asarray(target_year_by_row)

            # Get the EULP data for a specific building for use in the regressions
            df_eulp_pred = ts_agg.loc[bldg_id].copy()

            # HVAC ELECTRICITY
            # Predict HVAC electricity energy consumption
            df_eulp_hvac = prediction(
                base_year, df_eulp_pred, sw_test_base, target_years, sw_test_target,
                'HVAC.elec', weather_base_df, weather_target_df, bldg_id, df_eulp_targ,
                target_year_by_row
            )

            # Error check: Add regressed annual HVAC.elec to df_meta for a bldg_id
            df_meta.loc[bldg_id, 'HVAC.elec'] = (
                df_eulp_hvac.loc[
                    (df_eulp_hvac['timestamp'] - pd.Timedelta(hours=1)).dt.year
                    == comparison_year,
                    'HVAC.elec'
                ].sum().round(6))

            # Rename columns, including `HVAC.elec` to match the building ID
            df_eulp_hvac.columns = ['timestamp_EST', f'{bldg_id}']

            # Set the timestamp column to the index
            df_eulp_hvac.set_index('timestamp_EST', inplace=True)

            # Append single building DataFrame to list of building DataFrames
            df_bldg.append(df_eulp_hvac)

            # NATURAL GAS
            # Predict natural gas heating energy consumption
            df_eulp_ng = prediction(
                base_year, df_eulp_pred, sw_test_base, target_years, sw_test_target,
                'natural_gas.heating.energy_consumption',
                weather_base_df, weather_target_df, bldg_id, df_eulp_targ,
                target_year_by_row
            )

            # Add regressed annual ng to df_meta for a bldg_id
            df_meta.loc[bldg_id, 'natural_gas.heating.energy_consumption'] = (
                df_eulp_ng.loc[
                    (df_eulp_ng['timestamp'] - pd.Timedelta(hours=1)).dt.year
                    == comparison_year,
                    'natural_gas.heating.energy_consumption'
                ].sum().round(6))

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

# Collapse bldg_id (county/sim-county) columns to county columns.
# First, create lookup to get county from df_meta
county_labels = df_meta.loc[df_eulp.columns, county].astype(str)
# Then group by county and sum energy use across all buildings in each county.
df_eulp = df_eulp.T.groupby(county_labels).sum().T

# Aggregate df_meta to county-level before diagnostics.
# Drop sim-county column to avoid string concatenation during groupby sum.
df_meta = df_meta.drop(columns=['in.as_simulated_nhgis_county_gisjoin'],
                       errors='ignore')
df_meta = (
    df_meta.groupby([county, 'in.county_name', 'in.state'], as_index=False)
    .sum()
    .set_index(county)
)

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

# Keep the first 8760 hourly rows in each model year
# (assign Jan 1 00:00 to the previous year via -1 hour shift).
model_year = (df_eulp.index - pd.Timedelta(hours=1)).year
df_eulp = df_eulp.groupby(model_year, group_keys=False).head(8760)

# Save concatenated energy consumption DataFrame (hour x bldg) to CSV file
df_eulp.to_csv(os.path.join(output_dir,
    f'{prefix}eulp_hvac_elec_MWh_upgrade{upgrade}_'
    f'{start_index:04}-{end_index:04}.csv'))

print('\nChunk done at:', dt.datetime.now())
print('Total time elapsed:', dt.datetime.now() - script_start_time)
