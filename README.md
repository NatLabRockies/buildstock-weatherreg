# ResStock/ComStock EULP Weather-Year Regression Pipeline Quickstart

> **OS-Agnostic:** This program is OS agnostic. For larger runs (e.g., full national runs), the user should pursue access to the High Performance Computing (HPC) environment. Use of the HPC is optional, but directions for its use are integrated into the following instructions.

---

## Accesses

- **AWS Account and Allocation** – Please reach out to the Stratus Cloud team ([Stratus Cloud: Home](https://stratus-cloud.thesource.nrel.gov/)) to acquire them (you may need to refresh the page to load it).
- **AWS ResStock/ComStock Sandbox Access (likely `resbldg`)** – Please contact Buildings (ResStock/ComStock) teams for access.
- **HPC (optional)** – Please reach out to the [HPC team](https://www.nlr.gov/hpc/) for an account and allocation.

---

## Environment Setup

### Install AWS CLI
- Follow the [AWS CLI installation instructions](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).
  - For example, on HPC:
    ```
    cd ~
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    ./aws/install -i "$HOME/local/aws-cli" -b "$HOME/local/bin"
    ```
    We then add $HOME/local/bin to $PATH so we can use "aws ..."
    ```
    echo 'export PATH="$HOME/local/bin:$PATH"' >> ~/.bashrc
    ```
    Although unrelated to AWS, add this flag on HPC to trigger HPC-specific logic when running (only do this on HPC!)
    ```
    echo 'export REEDS_USE_SLURM=1' >> ~/.bashrc
    ```
    Finally, reload .bashrc:
    ```
    source ~/.bashrc
    ```
    And we confirm AWS installation and $PATH changes by seeing a version output from:
    ``` 
    aws --version
    ```
### AWS SSO Configuration
- Follow [AWS SSO configuration instructions](https://github.com/NatLabRockies/buildstock-query/wiki/AWS-setup#with-sso-for-nrel-employees), all the way through step 8! Note that after step 5 a browser should automatically open. We had luck using remote SSH from vscode, and didn't have luck with git bash or direct terminal sshing into kestrel. 

### Clone Repo
```bash
# Via HTTPS
git clone https://github.com/NatLabRockies/buildstock-weatherreg.git

# Via SSH (requires SSH key setup)
git clone git@github.com:NatLabRockies/buildstock-weatherreg.git
```

### Install the Geothermal Environment
`geotheraml_env_kestrel.yml` was created on and intended to be installed on Kestrel, while `geothermal_env.yml` was created on Windows and likely is more appropriate for Windows machines.

On Kestrel:
```bash
cd buildstock-weatherreg
conda env create -f geothermal_env_kestrel.yml
conda activate geothermal
```

This step can take ~30 minutes.

Then, copy the ComStock schema:
```bash
python -c "import site; print(site.getsitepackages())"
```
Locate `comstock_oedi.toml` in the current directory and copy it to:
```
<site-packages-dir>/buildstock_query/db_schema/
```

---

## Running the Program

```bash
conda activate geothermal
aws sso login
```

### Adjust Configuration
- Edit `switches_agg.json` to match your desired settings.
  - Ensure that `"workgroup"` matches your Stratus Cloud Handle / AWS Sandbox Workgroup.
- Edit `#SBATCH` settings at the top of `C_run_bldg_chunk_agg.sh` as needed.

### Test run
By default, `switches_agg.json` has `"testmode": true`, which activates a reduced ResStock run of Vermont only (see `'VT'` in `B_building_stock_parallel_agg.py`). Simply run:
```bash
python B_building_stock_parallel_agg.py
```

### National Runs
For national runs, it's best to launch `A_start_building_stock_parallel_agg.sh` instead of `B_building_stock_parallel_agg.py` so that the job can be queued and fully run on compute nodes.
- Edit `#SBATCH` settings at the top of  `A_start_building_stock_parallel_agg.sh`.
- In `switches_agg.json`:
  - Deactivate `testmode` by setting to `false`.
  - Select upgrades ("measures") to run with `upgrades` switch, which accepts a list of integers specifying the upgrades to run.
  - Select target years to run with `target_year` switch, which can either be an integer or a list of either integers or strings with ranges, inclusive of the end (e.g. `["2007-2013","2016-2023"]`).

#### ComStock
ResStock is used by default. For ComStock regressions, change these switches:
- `"comstock": true,`
- `"base_run": "comstock_2025_2"`
- `"chunk_size": 10`

#### Non-regressed
To simply pull existing ComStock or ResStock results, rather than running any regressions, change these switches:
- `"apply_regression": false`
- `"target_year"` must be set equal to `"base_year"`
- Set `"chunk_size"` to `500` for ResStock and `50` for ComStock.

## Troubleshooting

### Yampa-specific AWS SSO Configuration/Login
We had issues with AWS SSO configuration on Yampa. One solution is to perform authentication on your local machine (required for each login):
- Sign in on your local machine using `aws sso login`.
- Replace the SSO token cache directory (~/.aws/sso/cache) on Yampa with the one on local.

## Validation
See regression validation outputs for resstock and comstock HVAC EULP here: https://drive.google.com/file/d/1qDy9DrraTP7Kkzk1i6_tDVStEf3fzrQn/view?usp=sharing

