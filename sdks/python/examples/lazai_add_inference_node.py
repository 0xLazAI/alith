from alith.lazai import Client

client = Client()
client.add_inference_node(
    "<replace with your wallet address>",
    "<replace with your inference url>",
    "<replace with your RSA public key or RSA public key base64 format>",
)
