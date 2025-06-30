from alith.lazai import ProofData, Client, ChainConfig

proof = ProofData(id=1, score=1, file_url="", proof_url="")
client = Client(ChainConfig.local())
client.add_proof(1, proof)
