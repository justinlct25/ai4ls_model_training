import pandas as pd

# Load the previously merged CSV file
merged_df = pd.read_csv('merged_biodiversity_results.csv')

# Load the additional CSV file
additional_df = pd.read_csv('PRJNA952168_metadata.csv')

# Check if 'Sample Name' and 'lat_lon' columns exist in the additional CSV file
if 'Sample Name' in additional_df.columns and 'lat_lon' in additional_df.columns:
    # Convert 'lat_lon' column to string
    additional_df['lat_lon'] = additional_df['lat_lon'].astype(str)

    # Convert 'SampleID' to uppercase before merging
    merged_df['SampleID'] = merged_df['SampleID'].str.upper()

    # Extract 'Sample Name' and 'lat_lon' columns from the additional CSV file
    geo_loc_columns = additional_df[['Sample Name', 'lat_lon']]

    # Remove underscores from the 'Sample Name' column
    geo_loc_columns['Sample Name'] = geo_loc_columns['Sample Name'].str.replace('_', '0')

    # Merge the 'lat_lon' column with the previously merged DataFrame
    final_merged_df = pd.merge(merged_df, geo_loc_columns, left_on='SampleID', right_on='Sample Name', how='left')

    # Drop the additional 'Sample Name' column
    final_merged_df = final_merged_df.drop('Sample Name', axis=1, errors='ignore')

    # Extract latitude and longitude
    # lat_lon_parts = final_merged_df['lat_lon'].str.split(expand=True)
    # final_merged_df['Latitude'] = lat_lon_parts[0].astype(float) * (-1 if 'S' in lat_lon_parts[1] else 1)
    # final_merged_df['Longitude'] = lat_lon_parts[3].astype(float) * (-1 if 'W' in lat_lon_parts[4] else 1)

    # Drop the additional 'lat_lon' column
    # final_merged_df = final_merged_df.drop('lat_lon', axis=1, errors='ignore')

    # Drop duplicate rows based on 'SampleID'
    final_merged_df = final_merged_df.drop_duplicates(subset=['SampleID'])

    # Export the final merged DataFrame to a new CSV file
    final_merged_df.to_csv('final_merged_results_with_geo.csv', index=False)

    print("Final merged results with 'lat_lon' column exported successfully.")
else:
    print("The 'Sample Name' and 'lat_lon' columns are not present in the additional CSV file.")
