import pandas as pd

# Read the original CSV file into a DataFrame
original_df = pd.read_csv('../LUCAS_dataset/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv')

# List of land-use classes to be considered as unmanaged
unmanaged_land_classes = ['Forestry', 'Semi-natural and natural areas not in use', 'Other abandoned areas']

# Create the 'un-/managed land' column based on the list
original_df['Un-/Managed_LU'] = 1
original_df.loc[original_df['LU1_Desc'].isin(unmanaged_land_classes), 'Un-/Managed_LU'] = 0

# Print the updated DataFrame
print(original_df)

# Save the updated DataFrame to a new CSV file
original_df.to_csv('LUCAS-SOIL-2018(managed-l).csv', index=False)
