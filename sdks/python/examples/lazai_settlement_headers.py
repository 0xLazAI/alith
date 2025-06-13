from alith.lazai import Client, recover_address

client = Client()
user = client.wallet.address
node = client.wallet.address
nonce = 1
headers = client.get_request_headers(node, 1)
print(
    "The signed request headers is",
    headers,
)
signature = headers["X-LazAI-Signature"]
assert user == recover_address(nonce, user, node, signature)
