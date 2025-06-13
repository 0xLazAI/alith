import requests

from alith.lazai import ProofRequest

proofRequest = ProofRequest(
    job_id=1,
    file_id=1,
    file_url="1",
    encryption_key="2",
    encryption_seed="3",
    proof_url=None,
    nonce=None,
)

response = requests.post("http://127.0.0.1:80/proof", json=proofRequest.model_dump())

if response.status_code == 200:
    try:
        print("Proof request sent successfully")
        print("Response data:", response.json())
    except requests.exceptions.JSONDecodeError:
        print("Server returned non-JSON content:", response.text)
else:
    print(f"Failed with status code {response.status_code}")
    print("Response content:", response.text)
