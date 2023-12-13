import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('../csv_processing/LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion).csv')

soil_attributes = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K', 'BD 0-10']  # Include 'BD' in soil attributes

# Define the target variable as the presence or absence of erosion
data['Erosion'] = data[['EROSION_SHEET', 'EROSION_RILL', 'EROSION_GULLY', 'EROSION_MASS', 'EROSION_DEP', 'EROSION_WIND']].max(axis=1)

# Replace NaN values with 0 in all columns
data.fillna(0, inplace=True)

# Replace values below LOD in specified columns
for column in soil_attributes:
    data[column] = data[column].replace(['< LOD', '<  LOD', '<0.0'], 0).astype(float)

# Filter rows where either the erosion column or the 'BD' column has non-missing values
data = data.dropna(subset=['BD 0-10'], how='any')

# Separate features (X) and target variable (y) for erosion prediction
X_erosion = data[soil_attributes]
y_erosion = data['Erosion']

# Split the data into training and testing sets
X_train_erosion, X_test_erosion, y_train_erosion, y_test_erosion = train_test_split(X_erosion, y_erosion, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_model_erosion = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_erosion.fit(X_train_erosion, y_train_erosion)

# Save the trained model to a file
joblib.dump(rf_model_erosion, 'rf_model_erosion_presence_with_BD.pkl')

# Make predictions on the test set
y_pred_erosion = rf_model_erosion.predict(X_test_erosion)

# Evaluate the model
accuracy_erosion = accuracy_score(y_test_erosion, y_pred_erosion)
classification_report_erosion = classification_report(y_test_erosion, y_pred_erosion)

# Print the results
print(f'Accuracy for Erosion Presence Prediction: {accuracy_erosion}')
print(f'Classification Report for Erosion Presence Prediction:\n{classification_report_erosion}')

# Save accuracy to a text file
accuracy_filename = 'erosion_presence_model_accuracy_with_BD.txt'
with open(accuracy_filename, 'w') as file:
    file.write(f'Accuracy for Erosion Presence Prediction: {accuracy_erosion}\n')
    file.write(f'Classification Report for Erosion Presence Prediction:\n{classification_report_erosion}')
