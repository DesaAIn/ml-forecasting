import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Create years from 1950 to 2023
years = np.arange(1950, 2024)

# Create a list to hold data
data_list = []

# Generate data for each year
for id, year in enumerate(years, start=1):
    # Create random values for income, financing, and spending
    pendapatan = np.random.randint(2000000000, 3000000000)  # Random income
    pembiayaan = np.random.randint(10000000, 20000000)  # Random financing
    belanja = pendapatan + pembiayaan + np.random.randint(50000000, 200000000)  # Total spending
    
    # Create random values for detailed spending
    bd_pemerintahan_desa = np.random.randint(400000000, 800000000)
    bd_pembangunan_desa = np.random.randint(400000000, 800000000)
    bd_pembinaan_kemasyarakatan = np.random.randint(30000000, 400000000)
    bd_pemberdayaan_masyarakat = np.random.randint(30000000, 400000000)
    bd_penanggulangan_bencana = np.random.randint(30000000, 200000000)

    # Append the data as a dictionary
    data_list.append({
        "id": id,
        "tahun": year,
        "pendapatan": pendapatan,
        "pembiayaan": pembiayaan,
        "belanja": belanja,
        "bd_pemerintahan_desa": bd_pemerintahan_desa,
        "bd_pembangunan_desa": bd_pembangunan_desa,
        "bd_pembinaan_kemasyarakatan": bd_pembinaan_kemasyarakatan,
        "bd_pemberdayaan_masyarakat": bd_pemberdayaan_masyarakat,
        "bd_penanggulangan_bencana": bd_penanggulangan_bencana
    })

# Convert the list to a DataFrame
data_df = pd.DataFrame(data_list)

# Save DataFrame to JSON file
data_json_path = 'data_dana_desa_format.json'
data_df.to_json(data_json_path, orient='records', lines=True)

# Display the generated DataFrame
data_df.head(), data_json_path
