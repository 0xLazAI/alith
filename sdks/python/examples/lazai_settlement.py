from alith.lazai import Client, ChainConfig

node = "0x1122330000000000000000000000000000000000"
client = Client(chain_config=ChainConfig.local())
if not client.get_inference_node(node)[-1]:
    client.add_inference_node(node, "url", "node pub key")
try:
    user = client.get_user(client.wallet.address)
    print("user info", user)
except Exception:
    client.add_user(100000)
client.deposit(1000000)
client.deposit_inference(node, 10)
print(
    "The inference account of user is",
    client.get_inference_account(client.wallet.address, node)[0],
)
