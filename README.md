[Note: Please see Quickstart.docx]

Version 1: Markdown format (similar but changed format from Quickstart.docx) - from Quickstart_Revised_USED.md

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
- Follow [AWS SSO configuration instructions](https://github.com/NatLabRockies/buildstock-query/wiki/AWS-setup#with-sso-for-nrel-employees).

### Clone Repo
```bash
# Via HTTPS
git clone https://github.nrel.gov/ReEDS/ReEDS-2.0.git

# Via SSH (requires SSH key setup)
git clone git@github.nrel.gov:ReEDS/ReEDS-2.0.git
```

### Install the Geothermal Environment
```bash
conda env create -f geothermal_env.yml
conda activate geothermal
conda install git-lfs
pip install tensorflow
pip install --upgrade "numpy<2.0"
```

### Buildstock Query
```bash
pip install git+https://github.com/NREL/buildstock-query#egg=buildstock-query[full]
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

### Yampa-specific Setup
For AWS SSO on Yampa, authentication must be performed on a different machine (e.g., local).
- Sign in on your local machine using `aws sso login`.
- Use WinSCP to:
  - Remove the cache directory on Yampa.
  - Copy over the SSO token cache directory from your local machine to Yampa.
![image](https://github.com/user-attachments/assets/83e1265e-02a9-4bf7-8e57-b23a34129b30)

### Adjust Configuration
- Edit `switches_agg.json` to match your desired settings.
- Ensure that `"workgroup"` matches your Stratus Cloud Handle / AWS Sandbox Workgroup.

### Run Aggregation
```bash
python B_building_stock_parallel_agg.py
```

---
