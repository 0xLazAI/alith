from alith.lazai import Client, ProofData

contributor = Client()
node = Client()
print("wallet address", contributor.wallet.address)
url = "https://example.com/okyes.txt"
file_id = contributor.get_file_id_by_url(url)
# File not found, add it
if file_id == 0:
    file_id = contributor.add_file(url)
print("file id", file_id)
node_url = node.get_node(node.wallet.address)[1]
if node_url:
    node.remove_node(node.wallet.address)
node.add_node(
    node.wallet.address,
    "https://example.com/node",
    "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
)
node_fee = 10
node.update_node_fee(node_fee)
contributor.request_proof(file_id, node_fee)
ids = contributor.file_job_ids(file_id)
print("file proof job ids", ids)
job_id = ids[-1]
job = contributor.get_job(job_id)
print("job id", job_id)
print("job", job)
node_info = contributor.get_node(job[-1])
print("node info", node_info)
node.complete_job(job_id)
print("job completed for id", job_id)
node.add_proof(file_id, ProofData(id=file_id, file_url="", proof_url=""))
print("proof added for file id", file_id)
contributor.request_reward(file_id)
print("reward requested for file id", file_id)
