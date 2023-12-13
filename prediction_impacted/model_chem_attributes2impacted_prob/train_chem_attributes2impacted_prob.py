import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

# Load the dataset
data = pd.read_csv('../../csv_processing/LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion)(out-standard).csv')

soil_attributes = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']

# Define the target variable as the presence or absence of out-of-standard attributes
data['Is_Any_Out_of_Standard'] = data[['pH_H2O_OS', 'OC_OS', 'CaCO3_OS', 'P_OS', 'N_OS', 'K_OS']].any(axis=1).astype(int)

# Replace NaN values with 0 in all columns
data.fillna(0, inplace=True)

# Replace values below LOD in specified columns
for column in soil_attributes:
    data[column] = data[column].replace(['< LOD', '<  LOD', '<0.0'], 0).astype(float)

# Filter rows where either the out-of-standard column has non-missing values
data = data.dropna(subset=['Is_Any_Out_of_Standard'], how='any')

# Separate features (X) and target variable (y) for prediction
X_soil = data[soil_attributes]
y_soil = data['Is_Any_Out_of_Standard']

# Split the data into training and testing sets
X_train_soil, X_test_soil, y_train_soil, y_test_soil = train_test_split(X_soil, y_soil, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_model_soil = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_soil.fit(X_train_soil, y_train_soil)

# Save the trained model to a file
joblib.dump(rf_model_soil, 'rf_model_soil_probability.pkl')

# Predict the probability of out-of-standard attributes for the test set
y_prob_soil = rf_model_soil.predict_proba(X_test_soil)[:, 1]

# Evaluate the model using ROC-AUC score
roc_auc_soil = roc_auc_score(y_test_soil, y_prob_soil)

# Print the results
print(f'ROC-AUC for Probability of Soil Out of Standard: {roc_auc_soil}')

accuracy_filename_soil = 'soil_prob_model_accuracy.txt'
with open(accuracy_filename_soil, 'w') as file:
    file.write(f'ROC-AUC for Probability of Soil Out of Standard: {roc_auc_soil}')
