from alith.lazai import Client
import requests

client = Client()
node = "0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194"
# 1. Register user wallet on LazAI and deposit fees
# Join the iDAO and deposit the query fees, node is the official LazAI iDAO contract address
try:
    client.get_user(client.wallet.address)
except Exception:
    client.add_user(1000000)
print(
    "The query account of user is",
    client.get_query_account(client.wallet.address, node)[0],
)
# 2. Request query with the settlement headers
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
