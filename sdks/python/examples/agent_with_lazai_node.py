from alith.lazai import Client
from alith.lazai.node import ProofRequest
import requests

contributor = Client()
print("wallet address", contributor.wallet.address)
url = "https://example.com/newfile123.txt"
file_id = contributor.get_file_id_by_url(url)
# File not found, add it
if file_id == 0:
    file_id = contributor.add_file(url)
print("file id", file_id)
node_fee = 10
contributor.request_proof(file_id, node_fee)
ids = contributor.file_job_ids(file_id)
print("file proof job ids", ids)
job_id = ids[-1]
job = contributor.get_job(job_id)
print("job id", job_id)
print("job", job)
node_info = contributor.get_node(job[-1])
print("node info", node_info)
node_url = node_info[1]
# Request proof from the node
proof_request = ProofRequest(
    job_id=job_id,
    file_id=file_id,
    file_url=url,
    encryption_key="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    proof_url=None,
)
response = requests.post(
    f"http://{node_url}/proof",
    json=proof_request.model_dump(),
)
if response.status_code == 200:
    print("Proof request sent successfully")
else:
    print("Failed to send proof request:", response.json())
contributor.request_reward(file_id)
print("reward requested for file id", file_id)
