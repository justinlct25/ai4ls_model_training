import pandas as pd
import joblib

# Load the saved model
loaded_model = joblib.load('rf_model_erosion.pkl')

# Separate features (X) for prediction
X_new_data = pd.DataFrame({
    'pH_H2O': [8.28],
    'EC': [20.82],
    'OC': [14.7],
    'CaCO3': [200],
    'P': [10.3],
    'N': [1.4],
    'K': [180.2]
})

# Make predictions using the loaded model
y_pred_new_data = loaded_model.predict(X_new_data)

# Display the predictions
print('Predicted Erosion Types for New Data:')
print(y_pred_new_data)
