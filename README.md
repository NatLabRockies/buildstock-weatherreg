# ReEDS-2.0 Geothermal Pipeline Quickstart

> ⚙️ **OS-Agnostic:** This pipeline is compatible with macOS, Linux (e.g., HPC/Yampa), and Windows. For full national runs, access to the High Performance Computing (HPC) environment is recommended (but optional).

---

## 🔑 Quickstart: Accesses

- **AWS Account and Allocation:**  
  Contact the Stratus Cloud team ([Stratus Cloud: Home](https://stratus.nrel.gov/)).

- **AWS ResStock/ComStock Sandbox Access:**  
  Typically the `resbldg` sandbox. Contact Anthony Fontanini for access.

- **HPC (optional):**  
  Reach out to the [HPC team](https://www.nrel.gov/computational-science/high-performance-computing.html) for an account and allocation.

---

## 🖥️ Quickstart: Environment Setup

### 1. Install AWS CLI

- Installation guide: [AWS CLI Installation](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
- On **Linux (e.g. Yampa/HPC)**:
  - Determine system architecture:  
    ```bash
    uname -m
    ```
  - Follow instructions in section:  
    _“You can install without sudo if you specify directories that you already have write permissions to…”_
  - If the default command gives a "Permission denied" error, use:  
    ```bash
    ./aws/install -i ./local/aws-cli -b ./local/bin
    ```
  - Add AWS CLI to your path by editing `~/.bashrc`.  
    Customize based on your version (e.g., 2.17.5).

### 2. Set Up AWS SSO Login

- Instructions: [AWS SSO Setup](https://docs.aws.amazon.com/cli/latest/userguide/sso-configure.html)

### 3. Clone the Repository

```bash
# Via HTTPS
git clone https://github.nrel.gov/ReEDS/ReEDS-2.0.git

# Or via SSH (requires SSH key setup)
git clone git@github.nrel.gov:ReEDS/ReEDS-2.0.git
```

### 4. Set Up the Geothermal Environment

```bash
# Open Git Bash or terminal
cd ReEDS-2.0

# Create and activate environment
conda env create -f geothermal_env.yml
conda activate geothermal

# Install dependencies
conda install git-lfs
pip install tensorflow
pip install --upgrade "numpy<2.0"

# Install buildstock-query
pip install git+https://github.com/NREL/buildstock-query#egg=buildstock-query[full]
```

### 5. Copy ComStock DB Schema

```bash
python -c "import site; print(site.getsitepackages())"
```

Copy `comstock_oedi.toml` (from this directory) to:
```
<site-packages-dir>/buildstock_query/db_schema/
```

---

## 🚀 Quickstart: Running the Program

1. Navigate to the root of the repository:
   ```bash
   cd ReEDS-2.0
   conda activate geothermal
   ```

2. Authenticate with AWS:
   ```bash
   aws sso login
   ```

3. **(Yampa Only)**  
   SSO must be done from a different machine (e.g., local):

   - Login locally using `aws sso login`
   - Copy credentials from your local machine to Yampa using `WinSCP`
     - Remove Yampa’s cache directory
     - Replace it with the locally generated SSO cache directory

4. Adjust `switches_agg.json` settings:
   - Make sure `"workgroup"` matches your Stratus Cloud Handle / AWS Sandbox Workgroup

5. Run the aggregation:
   ```bash
   python B_building_stock_parallel_agg.py
   ```

---

## ✅ Final Notes

- HPC is optional, but highly recommended for large/national runs.
- Keep dependencies (e.g., `tensorflow`, `numpy`, `git-lfs`) up to date within the geothermal conda environment.
