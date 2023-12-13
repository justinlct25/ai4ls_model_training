import pandas as pd

# Assuming your CSV file is named 'your_file.csv'
csv_file_path = 'LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion)(out-standard)(textural-info).csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Define the conditions for the new column 'is_erosion'
conditions = (
    (df['EROSION_SHEET'] == 1) |
    (df['EROSION_RILL'] == 1) |
    (df['EROSION_GULLY'] == 1) |
    (df['EROSION_MASS'] == 1) |
    (df['EROSION_DEP'] == 1) |
    (df['EROSION_WIND'] == 1) |
    (df['EROSION_RILLGULLY'] == 1)
)

# Create the new column 'is_erosion'
df['EROSION_PRESENT'] = conditions.astype(int)

# Drop the original erosion-related columns
columns_to_drop = ['EROSION_SHEET', 'EROSION_RILL', 'EROSION_GULLY', 'EROSION_MASS', 'EROSION_DEP', 'EROSION_WIND', 'EROSION_RILLGULLY']
df.drop(columns=columns_to_drop, inplace=True)

# Save the updated DataFrame to a new CSV file if needed
df.to_csv('LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion-present)(out-standard)(textural-info).csv', index=False)
