import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib  


data = pd.read_csv('./LUCAS-SOIL-2018(managed-l).csv')

soil_attributes = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']

# Replace NaN values with 0 in all columns
data.fillna(0, inplace=True)

# Replace values below LOD in specified columns
for column in soil_attributes:
    data[column] = data[column].replace(['< LOD', '<  LOD', '<0.0'], 0).astype(float)

# Separate features (X) and target variable (y) for Task 1
X_task1 = data[soil_attributes]
y_task1 = data['Un-/Managed_LU']

# Split the data into training and testing sets
X_train_task1, X_test_task1, y_train_task1, y_test_task1 = train_test_split(X_task1, y_task1, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model_task1 = SVC(kernel='linear', verbose=True)
svm_model_task1.fit(X_train_task1, y_train_task1)

# Save the trained model to a file
joblib.dump(svm_model_task1, 'svm_model_attributes_managed_l.pkl')

# Make predictions on the test set
y_pred_task1 = svm_model_task1.predict(X_test_task1)

# Evaluate the model
accuracy_task1 = accuracy_score(y_test_task1, y_pred_task1)
classification_report_task1 = classification_report(y_test_task1, y_pred_task1)

# Print the results
print(f'Accuracy for Task 1: {accuracy_task1}')
print(f'Classification Report for Task 1:\n{classification_report_task1}')