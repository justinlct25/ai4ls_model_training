import os
import pandas as pd

# Specify the folder containing the TSV files
folder_path = './otus'

# Create an empty DataFrame to store the results
otu_results_df = pd.DataFrame(columns=['SampleID', 'UniqueOTUs', 'TotalOTUs'])

# Loop through all TSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tsv'):
        # Construct the full path to the TSV file
        file_path = os.path.join(folder_path, filename)

        # Load the TSV file into a pandas DataFrame
        df = pd.read_csv(file_path, delimiter='\t')

        # Extract the sample ID from the second column name
        sample_id = df.columns[1]

        # Calculate the number of unique OTUs
        unique_otus = df[df['Classification'] != 'No hits']['OTU'].nunique()

        # Calculate the total number of OTUs
        total_otus = len(df)

        # Append the results to the DataFrame
        otu_results_df = otu_results_df.append({
            'SampleID': sample_id,
            'UniqueOTUs': unique_otus,
            'TotalOTUs': total_otus
        }, ignore_index=True)

# Export the results DataFrame to a new CSV file
otu_results_df.to_csv('otu_results.csv', index=False)

print("OTU results calculated and exported successfully.")
