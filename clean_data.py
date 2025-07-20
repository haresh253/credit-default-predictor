import pandas as pd
import os

# Load the raw Excel file
xls_path = "default of credit card clients.xls"
df_raw = pd.read_excel(xls_path, header=1)

# Rename target column
df_raw.rename(columns={"default payment next month": "default"}, inplace=True)

# Save cleaned version to CSV
os.makedirs("data", exist_ok=True)
df_raw.to_csv("data/cleaned_data.csv", index=False)

print("✅ Cleaned data saved to data/cleaned_data.csv")
