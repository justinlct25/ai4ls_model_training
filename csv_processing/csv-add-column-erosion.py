import pandas as pd

original_df = pd.read_csv('./LUCAS-SOIL-2018(managed-l)(bulk-density).csv')
erosion_df = pd.read_csv('./LUCAS2018_EROSION.csv')

erosion_types = ['SHEET', 'RILL', 'GULLY', 'MASS', 'DEP', 'WIND']
directions = ['P', 'N', 'E', 'S', 'W']

# Iterate over erosion types and columns
for erosion_type in erosion_types:
    erosion_type_columns = [f'SURVEY_EROSION_{erosion_type}_{direction}' for direction in directions]
    new_column_name = f'EROSION_{erosion_type}'
    erosion_df[new_column_name] = (erosion_df[erosion_type_columns] == 1).any(axis=1).astype(int)

erosion_rill_gully_column = 'SURVEY_EROSION_RILLGULLY_N'
erosion_df['EROSION_RILLGULLY'] = erosion_df[erosion_rill_gully_column].notna().astype(int)

# Extract relevant columns from the erosion data for merging
erosion_columns_to_merge = ['POINT_ID'] + [f'EROSION_{erosion_type}' for erosion_type in erosion_types] + ['EROSION_RILLGULLY']
erosion_data = erosion_df[erosion_columns_to_merge]

# Merge the original dataset with the erosion data
merged_df = pd.merge(original_df, erosion_data, left_on='POINTID', right_on='POINT_ID', how='left')

# Drop the duplicate POINT_ID column
merged_df = merged_df.drop(columns=['POINT_ID'])

# Save the merged dataframe to a new CSV file
merged_df.to_csv('./LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion).csv', index=False)