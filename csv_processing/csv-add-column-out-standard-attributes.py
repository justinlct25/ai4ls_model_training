import pandas as pd
import sys


if len(sys.argv) != 2:
    print("Usage: python script.py <original_csv_file>")
    sys.exit(1)

original_csv_file = sys.argv[1]

# Read the original CSV file into a DataFrame
df = pd.read_csv(original_csv_file)

# Define sustainable standards for soil chemical attributes
standards = {
    'pH_H2O': {'min': 5.5, 'max': 8.5},
    # 'EC': {'min': 0.1, 'max': 1.0},
    'OC': {'min': 3.0, 'max': 20.0},
    # 'CaCO3': {'min': 0.0, 'max': 10.0},
    'P': {'min': 16.0, 'max': 140.0},
    'N': {'min': 1.5, 'max': float('inf')},
    'K': {'min': 121.0, 'max': 1500.0}
}

# Add columns indicating whether each attribute is out of standard
for attribute, values in standards.items():
    df[attribute] = df[attribute].replace(['< LOD', '<  LOD', '<0.0'], 0).astype(float)
    min_value = values['min']
    max_value = values['max']
    # Create a new column for each attribute indicating whether it's out of standard
    column_name = f'{attribute}_OS'
    df[column_name] = ((df[attribute] < min_value) | (df[attribute] > max_value)).astype(int)

df['Total_OS'] = df[[f'{attribute}_OS' for attribute in standards]].sum(axis=1)


# Print the updated DataFrame
print(df)

# Save the updated DataFrame to a new CSV file
output_csv_file = original_csv_file.replace('.csv', '(out-standard).csv')
df.to_csv(output_csv_file, index=False)
