import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
print("SCALER_MEAN =", list(np.round(scaler.mean_, 4)))
print("SCALER_STD  =", list(np.round(np.sqrt(scaler.var_), 4)))