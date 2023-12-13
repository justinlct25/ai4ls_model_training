import pandas as pd

# Load the diversity results file
diversity_df = pd.read_csv('diversity_results.csv')

# Load the OTU results file
otu_df = pd.read_csv('otu_results.csv')

# Merge the two DataFrames based on the 'SampleID' column
merged_df = pd.merge(diversity_df, otu_df, on='SampleID')

# Drop the 'file' column
merged_df = merged_df.drop('File', axis=1, errors='ignore')

# Export the merged DataFrame to a new CSV file
merged_df.to_csv('merged_biodiversity_results.csv', index=False)

print("Results merged and exported successfully.")
