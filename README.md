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
    source ~/.bashrc
    ```
    And we confirm installation and $PATH changes by seeing a version output from:
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
```bash
cd buildstock-weatherreg
conda env create -f geothermal_env.yml
```

You may experience errors during this step, but continue through the following steps, which might correct them.

```bash
conda activate geothermal
conda install git-lfs
pip install tensorflow
pip install --upgrade "numpy<2.0"
pip install "buildstock-query[full] @ git+https://github.com/NatLabRockies/buildstock-query@2024.05.09"
```

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
- Edit `#SBATCH` settings at the top of `A_start_building_stock_parallel_agg.sh` and `C_run_bldg_chunk_agg.sh` as needed.

### Run Aggregation
```bash
python B_building_stock_parallel_agg.py
```

## Troubleshooting

### Yampa-specific AWS SSO Configuration/Login
We had issues with AWS SSO configuration on Yampa. One solution is to perform authentication on your local machine (required for each login):
- Sign in on your local machine using `aws sso login`.
- Replace the SSO token cache directory (~/.aws/sso/cache) on Yampa with the one on local.

## Validation
See regression validation outputs for resstock and comstock HVAC EULP here: https://drive.google.com/file/d/1qDy9DrraTP7Kkzk1i6_tDVStEf3fzrQn/view?usp=sharing

