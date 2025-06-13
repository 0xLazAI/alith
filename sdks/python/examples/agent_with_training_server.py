from alith.training import run

"""Run the server and use the following command to test the server

curl -X 'POST' \
  'http://localhost:8000/v1/training' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "Qwen/Qwen2-0.5B",
  "template": "qwen",
  "learning_rate": 0.0001,
  "num_epochs": 3,
  "batch_size": 16
}'
"""
server = run()
