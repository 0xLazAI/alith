from alith.lazai import Client

address = "<replace with your wallet address>"
client = Client()
client.add_query_node(
    address,
    "<replace with your query node url>",
    "<replace with your RSA public key or RSA public key base64 format>",
)
print(client.get_query_node(address))
