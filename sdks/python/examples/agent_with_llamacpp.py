from alith.inference import LlamaEngine

engine = LlamaEngine("/root/models/qwen2.5-1.5b-instruct-q5_k_m.gguf", verbose=False)
print(engine.prompt("Calculate 10 - 3"))
