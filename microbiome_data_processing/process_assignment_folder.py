import os
import pandas as pd
import numpy as np

# Specify the folder containing the TSV files
folder_path = './assignments'

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['SampleID', 'ShannonDiversity', 'SimpsonDiversity'])

# Loop through all TSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tsv'):
        # Extract the sample ID from the filename (assuming the format is "LucasXXXX.tsv")
        file = filename.split('.')[0]


        # Construct the full path to the TSV file
        file_path = os.path.join(folder_path, filename)

        # Load the TSV file into a pandas DataFrame
        df = pd.read_csv(file_path, delimiter='\t')
        
        sample_id = df.columns[3]

        # Assuming the abundance column is the third column
        abundance_column = df.columns[3]

        # Calculate relative abundances
        total_abundance = df[abundance_column].sum()
        df['RelativeAbundance'] = df[abundance_column] / total_abundance

        # Calculate Shannon diversity index
        shannon_diversity = -np.sum(df['RelativeAbundance'] * np.log(df['RelativeAbundance']))

        # Calculate Simpson diversity index
        simpson_diversity = 1 / np.sum(df['RelativeAbundance']**2)

        # Append the results to the DataFrame
        results_df = results_df.append({
            'SampleID': sample_id,
            'ShannonDiversity': shannon_diversity,
            'SimpsonDiversity': simpson_diversity,
            'File': file
        }, ignore_index=True)

# Export the results DataFrame to a new CSV file
results_df.to_csv('diversity_results.csv', index=False)

print("Diversity indices calculated and exported successfully.")
