import pandas as pd
import numpy as np

# Load the TSV file into a pandas DataFrame
df = pd.read_csv('All_Assignments(SRR25384251).tsv', delimiter='\t')

# Assuming the abundance column is 'Lucas0371'
abundance_column = 'Lucas0371'

# Calculate relative abundances
total_abundance = df[abundance_column].sum()
df['RelativeAbundance'] = df[abundance_column] / total_abundance

# Calculate Shannon diversity index
shannon_diversity = -np.sum(df['RelativeAbundance'] * np.log(df['RelativeAbundance']))

# Calculate Simpson diversity index
simpson_diversity = 1 / np.sum(df['RelativeAbundance']**2)

print(f'Shannon Diversity Index: {shannon_diversity}')
print(f'Simpson Diversity Index: {simpson_diversity}')

