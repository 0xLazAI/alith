from alith import Agent, LazAIClient

LAZAI_IDAO_ADDRESS = "0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194"

client = LazAIClient()
# 1. Register user wallet on LazAI and deposit fees (Only Once)
# Join the iDAO and deposit the inference fees, node is the official LazAI iDAO contract address
try:
    client.get_user(client.wallet.address)
except Exception:
    client.add_user(1000000)
    client.deposit_inference(LAZAI_IDAO_ADDRESS, 100000)

# 2. Request the inference server with the settlement headers
url = client.get_inference_node(LAZAI_IDAO_ADDRESS)[1]
agent = Agent(
    base_url=f"{url}/v1", extra_headers=client.get_request_headers(LAZAI_IDAO_ADDRESS, file_id=1)
)
print(agent.prompt("What is Alith?"))
