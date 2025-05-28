from alith.lazai import Client, ProofRequest
from alith.data import encrypt
from alith.data.storage import (
    PinataIPFS,
    UploadOptions,
    GetShareLinkOptions,
    StorageError,
)
from eth_account.messages import encode_defunct
from os import getenv
import asyncio
import requests
import rsa


async def main():
    client = Client()
    ipfs = PinataIPFS()
    try:
        # 1. Prepare your privacy data and encrypt it
        data_file_name = "your_encrypted_data.txt"
        privacy_data = "Your Privacy Data"
        encryption_seed = "Sign to retrieve your encryption key"
        message = encode_defunct(text=encryption_seed)
        password = client.wallet.sign_message(message).signature.hex()
        encrypted_data = encrypt(privacy_data.encode(), password)
        # 2. Upload the privacy data to IPFS and get the shared url
        token = getenv("IPFS_JWT", "")
        file_meta = await ipfs.upload(
            UploadOptions(name=data_file_name, data=encrypted_data, token=token)
        )
        url = await ipfs.get_share_link(
            GetShareLinkOptions(token=token, id=file_meta.id)
        )
        # 3. Upload the privacy url to LazAI
        file_id = client.get_file_id_by_url(url)
        if file_id == 0:
            file_id = client.add_file(url)
        # 4. Request proof in the verified computing node
        client.request_proof(file_id, 100)
        job_id = client.file_job_ids(file_id)[-1]
        job = client.get_job(job_id)
        node_info = client.get_node(job[-1])
        node_url: str = node_info[1]
        pub_key = node_info[-1]
        encryption_key = rsa.encrypt(
            password.encode(),
            rsa.PublicKey.load_pkcs1(pub_key.strip().encode(), format="PEM"),
        ).hex()
        response = requests.post(
            f"{node_url}/proof",
            json=ProofRequest(
                job_id=job_id,
                file_id=file_id,
                file_url=url,
                encryption_key=encryption_key,
                encryption_seed=encryption_seed,
                proof_url=None,
            ).model_dump_json(),
        )
        if response.status_code == 200:
            print("Proof request sent successfully")
        else:
            print("Failed to send proof request:", response.json())
        # 5. Request DAT reward
        client.request_reward(file_id)
        print("Reward requested for file id", file_id)
        print(
            "The balance of DAT is",
            client.get_dat_balance(client.wallet.address, file_id),
        )
    except StorageError as e:
        print(f"Error: {e}")
    except Exception as e:
        raise e
    finally:
        await ipfs.close()


if __name__ == "__main__":
    asyncio.run(main())
