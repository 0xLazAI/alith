from alith.lazai import Client
import requests

client = Client()
node = "0x3e186f0b9568bf5415854577D042843A9f1C2266"
# 1. Register user wallet on LazAI and deposit fees
# Join the iDAO and deposit the query fees, node is the official LazAI iDAO contract address
try:
    client.get_user(client.wallet.address)
except Exception:
    client.add_user(1000000)
# 2. Deposit for the query account.
client.deposit_query(node, 500000)
print(
    "The query account of user is",
    client.get_query_account(client.wallet.address, node)[0],
)
# 3. Request query with the settlement headers
url = client.get_query_node(node)[1]
headers = client.get_request_headers(node)
print("request headers:", headers)
print(
    "request result:",
    requests.post(
        f"{url}/query/rag",
        headers=headers,
        json={
            "file_id": "1",
            "query": "What is Alith?",
        },
    ).json(),
)
