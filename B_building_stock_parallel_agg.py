import os
import multiprocessing
import sys
import shutil
import datetime as dt
import subprocess
import pandas as pd
import numpy as np
import json

# Helper function for running parallelized tasks via multiprocessing
def run_task(cmd):
    print(f'Running command: {cmd} at {dt.datetime.now()}\n')
    result = subprocess.run(cmd, shell=True, env=os.environ.copy(), check=True,
                            stdout=sys.stdout.buffer, stderr=sys.stderr.buffer)
    return result.returncode

if __name__ == "__main__":
    # Detect if running on HPC
    hpc = bool(int(os.environ.get('REEDS_USE_SLURM', 0)))
    print(f"Running on HPC: {hpc}")

    if not hpc:
        # Detect available CPU cores dynamically
        num_cores = max(2, os.cpu_count() - 2)  # Leave 2 cores free
        print(f"Using {num_cores} parallel processes")

    # Create outputs directory & copy input files to `/inputs` subdirectory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    time = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = f'{script_dir}/outputs/outputs_{time}'

    ## Function to exclude 'outputs' directory (prevents recursion)
    def exclude_dir(_dirname, filenames):
        return [name for name in filenames if name == 'outputs']

    ## Create & copy files to the output directory, excluding 'outputs'
    shutil.copytree(script_dir, f'{output_dir}/inputs', ignore=exclude_dir)

    ## Write the commit hash to a text file in the output folder
    with open(f'{output_dir}/inputs/commit_hash.txt', 'w') as f:
        f.write(subprocess.check_output(['git', 'rev-parse', 'HEAD'])
                        .strip()
                        .decode('utf-8'))


    # SWITCHES #TODO: Only import necessary for this script & reorder
    with open(os.path.join(script_dir, 'switches_agg.json'), 'r') as f:
        switch = json.load(f)
    ## Switch that designates comstock or resstock data
    sw_comstock = switch['comstock'] # if `False`, then resstock
    ## Note: resstock upgrades do not correspond to the same # as comstock
    upgrades = switch['upgrades'] # default: comstock = [0, 1, 18], resstock = [0, 1, 5]
    n_bldngs = switch['n_bldngs'] # 'all' for all buildings, 'assign' for assigned building id list from csv
    base_year = switch['base_year'] # Base year for the building stock
    target_year = switch['target_year'] # Target year for the building stock
    comstock_year, comstock_release = switch['version_comstock'][0], switch['version_comstock'][1]
    resstock_year, resstock_release = switch['version_resstock'][0], switch['version_resstock'][1]
    chunk_size = switch['chunk_size'] # number of combinations to pull at a time
    bsq_cols = switch['com_bsq_cols'] if sw_comstock else switch['res_bsq_cols'] # columns to group by
    applied_only = switch['applied_only'] # if `True`, only buildings with upgrade applied

    # URLs
    url_base = switch['url_base']
    url_comstock = f'{url_base}{comstock_year}/comstock_amy{base_year}_release_{comstock_release}/'
    url_resstock = f'{url_base}{resstock_year}/resstock_amy{base_year}_release_{resstock_release}/'
    url_bldg = url_comstock if sw_comstock else url_resstock

    # Load the state_county_map outside the loop
    state_county_map = pd.read_csv(
        f"{url_base}2025/comstock_amy2018_release_2/geographic_information/"
        "spatial_tract_lookup_table_publish_v8+1.csv",
        storage_options={"anon": True}
    )

    # Subset the DataFrame to include only the specified columns
    state_county_map = state_county_map[
        ["nhgis_county_gisjoin", "resstock_county_id", "state_abbreviation"]
    ]

    # Save the subsetted DataFrame as a CSV to the output directory
    state_county_map.to_csv(
        os.path.join(output_dir, "inputs", "spatial_tract_lookup_table.csv"),
        index=False
    )

    # MAIN
    for upgrade in upgrades: 
        if sw_comstock and comstock_year == "2025" and comstock_release == "2":
            print("Using custom metadata load logic for ComStock 2025 Release 2")

            # state_meta = state_county_map["state_abbreviation"].unique().tolist() # Can be used to filter by state (see line below)
            state_meta = ['CA', 'OR']  # For testing purposes, only Vermont is used

            all_meta = []
            for state in state_meta:
                state_county_map_iter = state_county_map[state_county_map["state_abbreviation"].isin([state])]
                county_meta = state_county_map_iter["nhgis_county_gisjoin"].unique().tolist()

                # county_meta = ['G5000030'] # TESTING
                for county in county_meta:
                    try:
                        url = (
                            f"{url_bldg}metadata_and_annual_results_aggregates/by_state_and_county/full/parquet/"
                            f"state%3D{state}/county%3D{county}/{state}_{county}_upgrade{upgrade}_agg.parquet"
                            # if upgrade > 0 else
                            # f"{url_bldg}metadata_and_annual_results_aggregates/by_state_and_county/full/parquet/"
                            # f"state%3D{state}/county%3D{county}/{state}_{county}_baseline_agg.parquet"
                        )
                        df = pd.read_parquet(url, storage_options={"anon": True})
                        df["in.state"] = state
                        df["in.nhgis_county_gisjoin"] = county
                        all_meta.append(df)
                    except Exception as e:
                        print(f"Failed to load metadata for {state}, {county}: {e}")

            if not all_meta:
                raise RuntimeError(f"No metadata loaded for upgrade {upgrade}.")

            df_meta = pd.concat(all_meta, ignore_index=True)
            # TODO: It doesn't make sense to filter by upgrade here - should be above `for upgrade in upgrades`
            df_meta = df_meta[df_meta["upgrade"] == upgrade]

            # Merge state_county_map w/ df_meta to bring in resstock_county_id
            df_meta = df_meta.merge(
                state_county_map[
                    ["nhgis_county_gisjoin", "resstock_county_id"]
                ].drop_duplicates(),
                how="left",
                left_on="in.nhgis_county_gisjoin",
                right_on="nhgis_county_gisjoin"
            )

            # Assign resstock_county_id to in.county_name
            df_meta["in.county_name"] = df_meta["resstock_county_id"]

        else:
            # Reformat metadata filepath based on upgrade number
            url_meta = f'metadata_and_annual_results/national/full/parquet/upgrade{upgrade}.parquet'

            # Read Parquet file into a DataFrame
            df_meta = pd.read_parquet(url_bldg + url_meta)

        # Remove Alaska and Hawaii
        df_meta = df_meta[~df_meta['in.state'].isin(['AK', 'HI'])]

        # Set `county` based on `sw_comstock` value
        county = 'in.nhgis_county_gisjoin' if sw_comstock else 'in.county'

        # TODO: Alter testing code blocks to incorporate ComStock 2025.2
        # TESTING - DELETE for production or comment out
        # Testing Subset 1
        ## For testing purposes, subset to Vermont state
        # df_meta = df_meta[df_meta['in.state'] == 'VT']
        
        ## For ResStock testing purposes, subset to 'Single-Family Detached'
        # df_meta = df_meta[df_meta['in.geometry_building_type_recs']
        #                           == 'Single-Family Detached']
        
        # For ComStock testing purposes, subset to 'LargeOffice'
        # df_meta = df_meta[df_meta['in.comstock_building_type'] == 'LargeOffice']

        # counties = df_meta[county].unique()[:1] # TODO: counties (or restrict) should be a switch
        
        ## For testing purposes, subset counties
        # Note: G5000030 is the NHGIS code for Bennington County, VT
        # G1901630 = Scott County, IA; G1901530 = Polk County, IA
        # df_meta = df_meta[df_meta[county].isin(['G5000030'])] # or next line
        # df_meta = df_meta[df_meta[county].isin(counties)]
        # Testing Subset 1 end

        # # Testing Subset 2
        # states = ['WY', 'VT', 'AK', 'ND', 'SD']

        # # Filter df_meta to only include the specified states
        # df_meta = df_meta[df_meta['in.state'].isin(states)]

        # # Group by state and select the first three counties from each state
        # df_meta_subset = df_meta.groupby('in.state').apply(lambda x: x[county].unique()[:3]).reset_index()

        # # Flatten the list of counties for each state
        # counties = [county for sublist in df_meta_subset[0].tolist() for county in sublist]

        # # Filter df_meta to only include the selected counties
        # df_meta = df_meta[df_meta[county].isin(counties)]
        # # Testing Subset 2 end

        # Restrict to buildings upgraded in the current upgrade iteration
        if applied_only:
            df_meta = df_meta[df_meta['applicability']]

        # Apply weight to sqft and natural gas energy consumption
        df_meta['in.sqft'] = df_meta['in.sqft..ft2'] * df_meta['weight']

        # Define elec_enduses based on whether it's ComStock or ResStock
        if sw_comstock:
            elec_enduses = [
                'out.electricity.heating.energy_consumption',
                'out.electricity.cooling.energy_consumption',
                'out.electricity.fans.energy_consumption',
                'out.electricity.heat_recovery.energy_consumption',
                'out.electricity.heat_rejection.energy_consumption',
                'out.electricity.pumps.energy_consumption'
            ]

        else:
            elec_enduses = [
                'out.electricity.heating.energy_consumption',
                'out.electricity.heating_fans_pumps.energy_consumption',
                'out.electricity.heating_hp_bkup.energy_consumption',
                'out.electricity.heating_hp_bkup_fa.energy_consumption',
                'out.electricity.cooling.energy_consumption',
                'out.electricity.cooling_fans_pumps.energy_consumption'
            ]

        elec_enduses = [item + '..kwh' for item in elec_enduses]

        gas_enduses = ['out.natural_gas.heating.energy_consumption']

        gas_enduses = [enduse + '..kwh' for enduse in gas_enduses]

        # Apply weight to energy consumption columns
        for enduse in elec_enduses + gas_enduses:
            df_meta[enduse] = df_meta[enduse] * df_meta['weight']

        # Create a new column for total electricity consumption
        df_meta['meta_HVAC.elec'] = df_meta[elec_enduses].sum(axis=1)

        # Create a new column for total natural gas consumption
        df_meta['meta_natural_gas.heating.energy_consumption'] = (
            df_meta[gas_enduses].sum(axis=1)
        )
        # Format groupby columns based on BuildStockQuery groups
        group_cols = [f'in.{col}' for col in bsq_cols]

        grouped = df_meta.groupby(group_cols, observed=True)
        df_meta = grouped.agg({
            'in.sqft': 'sum',
            'meta_HVAC.elec': 'sum',
            'meta_natural_gas.heating.energy_consumption': 'sum'
        }).reset_index()

        # Error check: Convert energy consumption columns from kWh to MWh & round
        df_meta['meta_HVAC.elec'] = (df_meta['meta_HVAC.elec'] / 1000).round(6)
        df_meta['meta_natural_gas.heating.energy_consumption'] = (
            df_meta['meta_natural_gas.heating.energy_consumption'] / 1000).round(6)

        df_meta['bldg_id'] = df_meta[group_cols].apply(tuple, axis=1).astype(str)
        df_meta.set_index('bldg_id', inplace=True)
        df_meta = df_meta.sort_values(by=[
            'in.state',
            'in.nhgis_county_gisjoin' if sw_comstock else 'in.county',
        ])

        # Save single upgrade DataFrame to CSV file
        prefix = 'com_' if sw_comstock else 'res_'
        meta_path = os.path.join(
            output_dir, f'{prefix}meta_master_upgrade{upgrade}.csv')
        df_meta.to_csv(meta_path)

        unique_counties = df_meta[county].unique()

        # Store all processes for non-HPC mode in `tasks` list
        try:
            tasks  # Check if it exists
        except NameError:
            tasks = []
        # Process counties in parallelized chunks
        for i in range(0, len(unique_counties), chunk_size):
            # Get the chunk of counties
            start_index = i
            end_index = i + chunk_size
            county_chunk = unique_counties[i:i + chunk_size]
            counties_str = '_'.join(county_chunk)
            # Submit job to HPC or run locally
            if hpc:
                # Call a shell script that creates a compute node and runs a python file
                os.system(
                    f'sbatch --job-name=chunk_{prefix}{upgrade}_{start_index}-{end_index} '
                    f'./C_run_bldg_chunk_agg.sh {start_index} {end_index} {meta_path} '
                    f'{upgrade} {prefix} {output_dir} {script_dir} {counties_str}'
                )

            else:
                # Store commands for multiprocessing
                cmd = (
                    f'python {output_dir}/inputs/D_process_chunk_agg.py '
                    f'{start_index} {end_index} {meta_path} {upgrade} {prefix} '
                    f'{output_dir} {script_dir} {counties_str}'
                )
                tasks.append(cmd)  # Collect tasks to run later

    # Run multiprocessing to execute all tasks in parallel
    if not hpc and tasks:
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_task, tasks)

    print("All chunks processed successfully!")
