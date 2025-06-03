from alith.inference import run

"""Run the server and run the following command to test the server

curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "your-model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}'
"""
server = run(model="/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf")
