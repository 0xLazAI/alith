from alith import Agent, LazAIClient

# 1. Join the iDAO, register user wallet on LazAI and deposit fees (Only Once)
LAZAI_IDAO_ADDRESS = "0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194"
client = LazAIClient()
try:
    client.get_user(client.wallet.address)
except Exception:
    client.add_user(10000000)
    client.deposit_inference(LAZAI_IDAO_ADDRESS, 1000000)

# 2. Request the inference server with the settlement headers and DAT file id
file_id = 1
url = client.get_inference_node(LAZAI_IDAO_ADDRESS)[1]
agent = Agent(
    # OpenAI-compatible inference server URL
    base_url=f"{url}/v1",
    # Extra headers for settlement and DAT file anchoring
    extra_headers=client.get_request_headers(LAZAI_IDAO_ADDRESS, file_id=file_id),
)
print(agent.prompt("What is Alith?"))
