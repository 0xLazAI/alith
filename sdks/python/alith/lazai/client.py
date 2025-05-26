from .contracts import (
    ContractConfig,
    DATA_REGISTRY_CONTRACT_ABI,
    VERIFIED_COMPUTING_CONTRACT_ABI,
)
from .chain import ChainConfig, ChainManager
from .proof import ProofData
from os import getenv
from web3 import Web3
from eth_account import Account
from hexbytes import HexBytes


class Client(ChainManager):
    """
    LazAI Client for interacting with the LazAI blockchain.

    This client provides methods to interact with the LazAI blockchain, including contract interactions
    and chain management.
    """

    def __init__(
        self,
        chain_config: ChainConfig = ChainConfig.local(),
        contract_config: ContractConfig = ContractConfig(),
        private_key: str = getenv("PRIVATE_KEY", ""),
    ):
        super().__init__(chain_config, private_key)
        self.contract_config = contract_config

        # Initialize contracts with their respective ABIs
        self.data_registry_contract = self.w3.eth.contract(
            address=contract_config.data_registry_address,
            abi=DATA_REGISTRY_CONTRACT_ABI,
        )
        self.verified_computing_contract = self.w3.eth.contract(
            address=contract_config.verified_computing_address,
            abi=VERIFIED_COMPUTING_CONTRACT_ABI,
        )

    def add_file(self, url: str) -> int:
        self.send_transaction(self.data_registry_contract.functions.addFile(url))
        return self.get_file_id_by_url(url)

    def get_file_id_by_url(self, url: str) -> int:
        """
        Get the file ID by its URL.

        Args:
            url (str): The URL of the file.

        Returns:
            int: The file ID associated with the URL.
        """
        return self.data_registry_contract.functions.getFileIdByUrl(url).call()

    def add_node(self, address: str, url: str, public_key: str):
        return self.send_transaction(
            self.verified_computing_contract.functions.addNode(address, url, public_key)
        )

    def remove_node(self, address: str):
        return self.send_transaction(
            self.verified_computing_contract.functions.removeNode(address)
        )

    def node_list(self):
        return self.verified_computing_contract.functions.nodeList().call()

    def get_node(self, addr: str):
        return self.verified_computing_contract.functions.getNode(addr).call()

    def update_node_fee(self, fee: int):
        return self.send_transaction(
            self.verified_computing_contract.functions.updateNodeFee(fee)
        )

    def request_proof(self, file_id: int, value: int = 0):
        return self.send_transaction(
            self.verified_computing_contract.functions.requestProof(file_id),
            value=value,
        )

    def complete_job(self, job_id: int):
        return self.send_transaction(
            self.verified_computing_contract.functions.completeJob(job_id)
        )

    def add_proof(self, file_id: int, data: ProofData):
        packed_data = data.abi_encode()
        message_hash = Web3.keccak(packed_data)
        eth_message = Web3.keccak(b"\x19Ethereum Signed Message:\n32" + message_hash)
        signed_message = Account.signHash(eth_message, self.wallet.key)
        signature = signed_message.signature

        proof = {
            "signature": HexBytes(signature).hex(),
            "data": {
                "id": data.id,
                "fileUrl": data.file_url,
                "proofUrl": data.proof_url,
            },
        }

        return self.send_transaction(
            self.data_registry_contract.functions.addProof(file_id, proof)
        )

    def get_job(self, job_id: int):
        return self.verified_computing_contract.functions.getJob(job_id).call()

    def get_node(self, addr: str):
        return self.verified_computing_contract.functions.getNode(addr).call()

    def file_job_ids(self, file_id: int):
        return self.verified_computing_contract.functions.fileJobIds(file_id).call()

    def request_reward(self, file_id: int, proof_index: int = 1):
        return self.send_transaction(
            self.data_registry_contract.functions.requestReward(file_id, proof_index)
        )
