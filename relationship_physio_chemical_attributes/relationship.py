import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../csv_processing/LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion)(out-standard)(textural-info).csv')

# Assuming your DataFrame is named 'data'
chemical_attributes = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']
physical_attributes = ['Coarse', 'Clay', 'Silt', 'BD 0-20']

# Select relevant columns for analysis
selected_columns = chemical_attributes + physical_attributes
correlation_matrix = data[selected_columns].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()
