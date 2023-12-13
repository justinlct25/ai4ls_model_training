import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib


data = pd.read_csv('../../csv_processing/LUCAS-SOIL-2018(managed-l)(bulk-density)(erosion).csv')


# Extract features (X) and target variable (y) for erosion prediction
features = data[["LU1_Desc", "LC0_Desc"]]
# features = data[["LU1_Desc", "LC0_Desc"]]
data['Erosion'] = data[['EROSION_SHEET', 'EROSION_RILL', 'EROSION_GULLY', 'EROSION_MASS', 'EROSION_DEP', 'EROSION_WIND']].max(axis=1)
target_erosion = data['Erosion']

data.fillna(0, inplace=True)

# Encode categorical variables
label_encoder_lu1 = LabelEncoder()
features["LU1_Desc_encoded"] = label_encoder_lu1.fit_transform(features["LU1_Desc"])
label_encoder_lc0 = LabelEncoder()
features["LC0_Desc_encoded"] = label_encoder_lc0.fit_transform(features["LC0_Desc"])
joblib.dump(label_encoder_lu1.classes_, "label_encoder_lu1_classes.pkl")
joblib.dump(label_encoder_lc0.classes_, "label_encoder_lc0_classes.pkl")

# Drop the original categorical columns
features = features.drop(["LU1_Desc", "LC0_Desc"], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target_erosion, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest classifier for erosion prediction
rf_model_erosion = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_erosion.fit(X_train_scaled, y_train)

# Save the trained model to a file
joblib.dump(rf_model_erosion, 'rf_model_erosion_probability.pkl')

# Predict the probability of erosion for the test set
y_prob_erosion = rf_model_erosion.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model using ROC-AUC score
roc_auc = roc_auc_score(y_test, y_prob_erosion)

# Print the results
print(f'ROC-AUC for Probability of Erosion: {roc_auc}')

accuracy_filename = 'erosion_prob_model_accuracy.txt'
with open(accuracy_filename, 'w') as file:
    file.write(f'ROC-AUC for Probability of Erosion: {roc_auc}')
