import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv('../../csv_processing/LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion)(out-standard).csv')

# Replace values below LOD in specified columns for both datasets
attribute_columns = ['pH_H2O', 'EC', 'OC', 'CaCO3', 'P', 'N', 'K']

data.fillna(0, inplace=True)
for column in attribute_columns:
    data[column] = data[column].replace(['< LOD', '<  LOD', '<0.0'], 0).astype(float)

# Count the number of out-of-standard attributes for each sample
data['Num_Out_of_Standard'] = data[['pH_H2O_OS', 'OC_OS', 'CaCO3_OS', 'P_OS', 'N_OS', 'K_OS']].sum(axis=1)
# data['Is_Out_of_Standard'] = data[['pH_H2O_OS', 'OC_OS', 'CaCO3_OS', 'P_OS', 'N_OS', 'K_OS']].any(axis=1)


# Extract features (X) and target variable (y) for general impact prediction
features = data[attribute_columns]
target = data['Num_Out_of_Standard']
# target = data["Is_Out_of_Standard"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVC model
svm_model_num_out_of_standard = SVC(kernel='linear', verbose=True, probability=True)
svm_model_num_out_of_standard.fit(X_train_scaled, y_train)

# Save the trained model to a file
joblib.dump(svm_model_num_out_of_standard, 'svm_model_num_out_of_standard_probability.pkl')

# Predict the probability of the number of out-of-standard attributes for the test set
y_prob_num_out_of_standard = svm_model_num_out_of_standard.predict_proba(X_test_scaled)

# Make predictions on the test set
y_pred_num_out_of_standard = svm_model_num_out_of_standard.predict(X_test_scaled)

# Evaluate the model
accuracy_num_out_of_standard = accuracy_score(y_test, y_pred_num_out_of_standard)
classification_report_num_out_of_standard = classification_report(y_test, y_pred_num_out_of_standard)

# Print the results
print(f'Accuracy for Number of Out-of-Standard Prediction: {accuracy_num_out_of_standard}')
print(f'Classification Report for Number of Out-of-Standard Prediction:\n{classification_report_num_out_of_standard}')

# Save the accuracy results to a text file
accuracy_filename = 'num_out_of_standard_prob_model_accuracy.txt'
with open(accuracy_filename, 'w') as file:
    file.write(f'Accuracy for Number of Out-of-Standard Prediction: {accuracy_num_out_of_standard}\n'
               f'Classification Report for Number of Out-of-Standard Prediction:\n{classification_report_num_out_of_standard}')
