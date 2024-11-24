import os
from flask import Flask
from flask import request
from flask import jsonify
import pickle as pkl
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

model_name = os.getenv("MODEL_NAME")

# Load model
with open(f"{model_name}.pkl", "rb") as f:
    model = pkl.load(f)

# Load Encoder
with open("encoder.bin", "rb") as f:
    encoder = pkl.load(f)

# Load Scaler
with open("scaler.bin", "rb") as f:
    scaler = pkl.load(f)
    
# Load dv
with open("dv.bin", "rb") as f:
    dv = pkl.load(f)

app = Flask('heart-failure')

def prepare_data(patient, encoder, scaler) -> pl.DataFrame:
    columns = patient.keys()
    patient = pl.DataFrame([patient])
    # Prepare
    patient = patient.with_columns(
        Sex = encoder.fit_transform(patient['Sex']),
        ChestPainType = encoder.fit_transform(patient['ChestPainType']),
        RestingECG = encoder.fit_transform(patient['RestingECG']),
        ExerciseAngina = encoder.fit_transform(patient['ExerciseAngina']),
        ST_Slope = encoder.fit_transform(patient['ST_Slope'])
    )
    # Scaler
    patient = scaler.fit_transform(patient) # Applying MinMaxScaler
    return pl.DataFrame(patient, schema=columns)

def predict_single(patient, encoder, scaler, dv, model):
    X = prepare_data(patient, encoder, scaler)
    X = X.to_dicts()
    X = dv.transform(X)
    X = csr_matrix(X)
    y_pred = model.predict(X)
    return y_pred

def predict_single_proba(patient, encoder, scaler, dv, model):
    X = prepare_data(patient, encoder, scaler)
    X = X.to_dicts()
    X = dv.transform(X)
    X = csr_matrix(X)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    prediction = predict_single_proba(patient, encoder, scaler, dv, model)
    failure = prediction >= 0.5

    result = {
        'heart_failure_probability': round(float(prediction), 3), ## we need to cast numpy float type to python native float type
        'heart_failure': bool(failure),  ## same as the line above, casting the value using bool method
    }

    return jsonify(result)  ## send back the data in json format to the user

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) 