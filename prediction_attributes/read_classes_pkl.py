import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
import joblib

classes1 = joblib.load("./model_lu_lc2chem_attributes/label_encoder_lu1_classes.pkl")
classes2 = joblib.load("./model_lu_lc2chem_attributes/label_encoder_lc0_classes.pkl")
print("chem: ")
print(classes1)
print(classes2)

classes1 = joblib.load("./model_lu_lc2phy_attributes_texture/label_encoder_lu1_classes.pkl")
classes2 = joblib.load("./model_lu_lc2phy_attributes_texture/label_encoder_lc0_classes.pkl")
print("texture: ")
print(classes1)
print(classes2)

classes1 = joblib.load("./model_lu_lc2phy_attributes_bulk/label_encoder_lu1_classes.pkl")
classes2 = joblib.load("./model_lu_lc2phy_attributes_bulk/label_encoder_lc0_classes.pkl")
print("bulk: ")
print(classes1)
print(classes2)