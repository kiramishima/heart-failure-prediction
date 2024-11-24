import json
import requests
from deepdiff import DeepDiff
import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)

model_name = os.getenv("MODEL_NAME")

def read_text(file):
    test_directory = Path(__file__).parent
    LOGGER.info(test_directory)
    with open(test_directory / file, 'rt', encoding='utf-8') as f_in:
        return json.load(f_in)
      
input = heart0 = read_text('sample.json')
    
url = 'http://localhost:9696/predict'
actual_response = requests.post(url, json=input).json()
print('actual response:')
print(json.dumps(actual_response, indent=2))

expected_response = {
  "heart_failure": True,
  "heart_failure_probability": 0.762
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff