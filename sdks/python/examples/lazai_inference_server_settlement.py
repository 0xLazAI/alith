from alith.inference import run

"""Run the server and use the following command to test the server

curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "X-LazAI-User: 0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194" \
-H "X-LazAI-Nonce: 123456" \
-H "X-LazAI-Signature: HSDGYUSDOWP123" \
-H "X-LazAI-Token-ID: 1" \
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
server = run(model="/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf", settlement=True)
