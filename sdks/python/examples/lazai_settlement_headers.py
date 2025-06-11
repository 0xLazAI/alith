from alith.lazai import Client

client = Client()
print(
    "The signed request headers is",
    client.get_request_headers("0xABCD", 1),
)
