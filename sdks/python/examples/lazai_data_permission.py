import asyncio

import rsa
from eth_account.messages import encode_defunct
from alith.lazai import Client


async def main():
    client = Client()
    try:
        # 1. Prepare your privacy data and encrypt it
        encryption_seed = "Sign to retrieve your encryption key"
        message = encode_defunct(text=encryption_seed)
        password = client.wallet.sign_message(message).signature.hex()
        file_url = "test url"
        # 3. Upload the privacy url to LazAI
        file_id = client.get_file_id_by_url(file_url)
        if file_id == 0:
            file_id = client.add_file(file_url)
        pub_key = r"""\
-----BEGIN RSA PUBLIC KEY-----
MIIBigKCAYEAgXskqGZXdIIsAvWi3AhLO4cStx4wCiWWK2kHL34M1B2ic3hE4PP6
VjUvcPz1loiDT0GhlrrvrUeWcJpElQrTAsuYPNmt8GCIec6n4LvEkIUfomLMsTJ0
tD16xb/xfv8F5Jo38cazNuoXN2X/knsQcWWbk2FTUsRETNb5kR6j1vcAWTCdyD+w
iuKZ6DqG0RSOnN0ES9NFTYa995GWxIobQWioh8U3hCyRwJ65C342IPuOoQJrMc9X
yx5jQiwisQfhbRj6wVOi1Qq9lROZGz5DaWtqgsB2/+BzMBV0ducdD72qcwr1hsN/
1xzQtEFnQTAZft1o41KOP/OxM98ezo1VV6BjIjHTcBAALhRqGTT5GtZ8RanFzkgK
yCu/GpUzYETOetm/Eio7pQo3WlTQtyXWZtnWvZb1394WxYQBryJG+h7YvN8rQv4S
ps7XUytVWo4Orjp4SoIkt3R0nr8kfMBhwncY1GnlrPi334cV46pCwFHNxO229Yb9
nqVggyRxv9s9AgMBAAE=
-----END RSA PUBLIC KEY-----
"""
        encryption_key = rsa.encrypt(
            password.encode(),
            rsa.PublicKey.load_pkcs1(pub_key.strip().encode(), format="PEM"),
        ).hex()
        client.add_permission_for_file(
            file_id,
            client.contract_config.data_registry_address,
            encryption_key,
        )
    except Exception as e:
        raise e
    finally:
        pass


if __name__ == "__main__":
    asyncio.run(main())
