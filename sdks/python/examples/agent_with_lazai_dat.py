from alith.lazai import Client

client = Client()
print("Wallet address", client.wallet.address)
print("Balance of DAT", client.get_dat_balance(client.wallet.address, 1))
