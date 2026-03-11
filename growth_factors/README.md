# Growth Factors
The files in this folder can be used to apply scaling factors to load profiles to approximate growth in the building stock over time. 

The following files are sourced from EIA AEO 2025:
- `Table_4._Residential_Sector_Key_Indicators_and_Consumption.csv`
    - Row 11, labeled `Residential: Key Indicators: Households: Total: Reference case` contains the number of dwelling units projected in each year. 
    - These values can be divided by the number of dwelling units represented in the ResStock yaml input file for the release being used [(example: 139,647,020 dwelling units for ResStock 2025.1)](https://github.com/NatLabRockies/resstock/blob/dfb702d505b9826ce4f0061073464c42ce9bbc5a/project_national/sdr_upgrades_tmy3.yml#L14) to derive scaling factors for each future year.
- `Table_5._Commercial_Sector_Key_Indicators_and_Consumption.csv`
    - Row 10, labeled `Commercial: Total Floorspace: Total: Reference case` contains the total commercial floorspace projected in each year. 
    - These values can be divided by the total commercial floorspace represented in ComStock (including the gap model that represents floorspace not explicitly modeled in ComStock). According to Eric Ringold, this value is based on CBECS 2018, so 96,423 million ft2 [(source)](https://www.eia.gov/consumption/commercial/data/2018/bc/pdf/b1.pdf). The gap model is based on 2018 data from EIA 930, so CBECS 2018 is the correct year to use for scaling. We are going to assume the gap model scales as the same rate as the ComStock portion of the commercial building stock. For future reference, CBECS data are used to derive weights by building type [here](https://github.com/NatLabRockies/ComStock/blob/aa3b988568b5963899836a7a762de990f5988043/postprocessing/comstockpostproc/comstock.py#L2088).
