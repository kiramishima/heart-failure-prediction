from pathlib import Path
import json
import os
from heart_failure.app import predict_single, predict_single_proba
import logging
import pickle as pkl

LOGGER = logging.getLogger(__name__)

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
    
def read_text(file):
    test_directory = Path(__file__).parent
    LOGGER.info(test_directory)
    with open(test_directory / file, 'rt', encoding='utf-8') as f_in:
        return json.load(f_in)

def test_predict_single():
    heart0 = read_text('sample.json')
    result = predict_single(heart0, encoder, scaler, dv, model)
    LOGGER.info(result)
    expected_result = [1]
    assert result == expected_result

def test_predict_single_proba():
    heart0 = read_text('sample.json')
    result = predict_single_proba(heart0, encoder, scaler, dv, model)
    LOGGER.info(result)
    expected_result = 0.7615042078687632
    assert result == expected_result