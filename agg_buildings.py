#!/usr/bin/env python
#SBATCH --account=geohc
#SBATCH --time=4:00:00
#SBATCH --mail-user=mmowers@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=246064    # RAM in MB; up to 246064 for normal or 2000000 for bigmem on kestrel
#SBATCH --qos=high

'''
This script develops county-level building hvac load profiles for ReEDS (in GWh) based on resstock/comstock EULP outputs (MWh). The resulting files are dropped back into the directories where the raw EULP files are located. This file can be run with python or via slurm with "sbatch agg_buildings.py".
'''

import os
import pandas as pd
import shutil
from pdb import set_trace as b

#Inputs
bldg_type = 'com' # 'com' for comstock or 'res' for resstock
bldg_path = r'/projects/geohc/geo_predict/outputs/outputs_2025-12-02-09-29-48'
bldg_tech = 'upgrade0'

shutil.copy2(os.path.realpath(__file__), bldg_path)
this_dir_path = os.path.dirname(os.path.realpath(__file__))

eulp_files = [f'{bldg_path}/{f}' for f in os.listdir(bldg_path) if f.startswith(f'{bldg_type}_eulp_hvac_elec_MWh_{bldg_tech}')]

ls_df = []
for f in eulp_files:
    print(f)
    df = pd.read_csv(f, index_col=0)
    counties = [c.replace('"',"'").replace("('","").replace("')","").split("', '")[0] for c in df.columns]
    cnty_fips = [int(c[1:3] + c[4:7]) for c in counties]
    df.columns = cnty_fips
    df_sum = df.groupby(df.columns, axis=1).sum()
    ls_df.append(df_sum)

df_sum = pd.concat(ls_df, axis=1)
df_sum = df_sum.groupby(df_sum.columns, axis=1).sum()
df_sum = df_sum/1000 #Convert to GWh
df_sum.to_csv(f'{bldg_path}/agg_{bldg_type}_eulp_hvac_elec_GWh_{bldg_tech}.csv')
