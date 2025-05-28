from alith.lazai import Client

node = Client()
print("wallet address", node.wallet.address)
node_url = node.get_node(node.wallet.address)[1]
if node_url:
    node.remove_node(node.wallet.address)
node.add_node(
    node.wallet.address,
    "https://f29d4e7ec6d010ff0aff401e21a3768ee9051935-8000.dstack-prod7.phala.network",
    """-----BEGIN RSA PUBLIC KEY-----
MIIBigKCAYEAjxlRv0gBLnzEly7X3pxkcdYhY6++ybWpQYoFkpjdGmUpoK6ULKgl
AOpwuWkek1Cnc/LWMNL1hj7qCeplK0xbwhEWHRExHiCsHpWv/pnYU2YY4ZN+1Scg
4/R/EnQ2h5Bnup9IMXpaBqQT7yr7ZWTPXNlOQwftvU5HjH61+fbDtZwN4Sra2ht4
JKtfNGA36uELu0krmchs9m/LK9o3hrH7bZUPcrtuTrHRkHI9QvQwBTEjsOiQDCtw
68JLay+lpScb2rR3NFCq6gBNjwLuZDyli7tSj+LHHxA8/J3PSfoyF/MIVojU0qYy
25qSLpig+YyQJ0x+zi6Gj0vBBlkFqdmY0gOaw0ZfeIbKC6gqIlVwifL6UeJVtEyq
Gn1y3ngjlSd3sbxqMDZHjPF4qTAmt3qvIKzRN/9IEpUSHJ1uzq5PbalD2W38xQ2v
3cF8PFUJ7w/vdGjRXwke1mxxPziJSBKrQIJYmg1vybMw+ii75ZsOiqdQr889huzZ
s+gWvVlFSYsrAgMBAAE=
-----END RSA PUBLIC KEY-----
""",
)
