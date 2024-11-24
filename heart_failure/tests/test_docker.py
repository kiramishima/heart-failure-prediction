import json
import requests
from deepdiff import DeepDiff

with open('sample.json', 'rt', encoding='utf-8') as f_in:
    input = json.load(f_in)
    
url = 'http://localhost:9696/predict'
actual_response = requests.post(url, json=input).json()
print('actual response:')
print(json.dumps(actual_response, indent=2))

expected_response = {
  "heart_failure": true,
  "heart_failure_probability": 0.762
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff