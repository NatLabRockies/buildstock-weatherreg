[Note: Please see Quickstart.docx]

Version 1: Markdown format (similar but changed format from Quickstart.docx) - from Quickstart_Revised_USED.md

# ReEDS/ComStock Geothermal Pipeline Quickstart

> ⚙️ **OS-Agnostic:** This program is OS agnostic. For larger runs (e.g., full national runs), the user should pursue access to the High Performance Computing (HPC) environment. Use of the HPC is optional, but directions for its use are integrated into the following instructions.

---

## 🔑 Quickstart: Accesses

- **AWS Account and Allocation** – Please reach out to the Stratus Cloud team ([Stratus Cloud: Home](https://stratus.nrel.gov/)) to acquire them.
- **AWS ResStock/ComStock Sandbox Access (likely `resbldg`)** – Please contact Anthony Fontanini for access.
- **HPC (optional)** – Please reach out to the [HPC team](https://www.nrel.gov/computational-science/high-performance-computing.html) for an account and allocation.

---

## 🛠️ Quickstart: Environment Setup

### Install AWS CLI
- Follow the [AWS CLI installation instructions](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).
- **On Linux (e.g. Yampa/HPC):**
  - Check architecture with:
    ```bash
    uname -m
    ```
  - Follow the section that begins with:
    _“You can install without sudo if you specify directories that you already have write permissions to…”_
    ![image](https://github.com/user-attachments/assets/54c0ac52-b57d-40bf-9384-b01053a8b259)
  - If the default install command fails with `Permission denied`, use:
    ```bash
    ./aws/install -i ./local/aws-cli -b ./local/bin
    ```
  - Add the AWS CLI path to your `~/.bashrc`. Customize based on the version installed (e.g., 2.17.5). Line 25 is required; Line 24 may not be necessary.
    ![image](https://github.com/user-attachments/assets/520db093-528a-41cf-bbf4-a5da43151cc6)

### AWS SSO Login Setup
- Follow [AWS SSO login setup instructions](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure.html).

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

## ▶️ Quickstart: Running the Program

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
