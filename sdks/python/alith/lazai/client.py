import random
import time
from os import getenv
from typing import Dict, List

from eth_account.messages import encode_defunct
from hexbytes import HexBytes
from web3 import Web3

from .chain import ChainConfig, ChainManager
from .contracts import (
    AI_PROCESS_CONTRACT_ABI,
    DATA_ANCHORING_TOKEN_CONTRACT_ABI,
    DATA_REGISTRY_CONTRACT_ABI,
    SETTLEMENT_CONTRACT_ABI,
    VERIFIED_COMPUTING_CONTRACT_ABI,
    ContractConfig,
)
from .proof import ProofData, SettlementData
from .settlement import SettlementRequest


class Client(ChainManager):
    """
    LazAI Client for interacting with the LazAI blockchain.

    This client provides methods to interact with the LazAI blockchain, including contract interactions
    and chain management.
    """

    def __init__(
        self,
        chain_config: ChainConfig = (
            ChainConfig.local()
            if getenv("LAZAI_LOCAL_CHAIN")
            else ChainConfig.testnet()
        ),
        contract_config: ContractConfig = (
            ContractConfig.local()
            if getenv("LAZAI_LOCAL_CHAIN")
            else ContractConfig.testnet()
        ),
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
        self.data_anchoring_token_contract = self.w3.eth.contract(
            address=contract_config.data_anchoring_token_address,
            abi=DATA_ANCHORING_TOKEN_CONTRACT_ABI,
        )
        self.query_contract = self.w3.eth.contract(
            address=contract_config.query_address,
            abi=AI_PROCESS_CONTRACT_ABI,
        )
        self.inference_contract = self.w3.eth.contract(
            address=contract_config.inference_address,
            abi=AI_PROCESS_CONTRACT_ABI,
        )
        self.training_contract = self.w3.eth.contract(
            address=contract_config.training_address,
            abi=AI_PROCESS_CONTRACT_ABI,
        )
        self.settlement_contract = self.w3.eth.contract(
            address=contract_config.settlement_address,
            abi=SETTLEMENT_CONTRACT_ABI,
        )

    def get_public_key(self) -> str:
        return self.data_registry_contract.functions.publicKey().call()

    def add_file(self, url: str) -> int:
        return self.add_file_with_hash(url, "")

    def add_file_with_hash(self, url: str, hash: str) -> int:
        self.send_transaction(self.data_registry_contract.functions.addFile(url, hash))
        return self.get_file_id_by_url(url)

    def add_permission_for_file(self, file_id: int, account: str, key: str):
        return self.send_transaction(
            self.data_registry_contract.functions.addPermissionForFile(
                file_id, account, key
            )
        )

    def get_file_id_by_url(self, url: str) -> int:
        """
        Get the file ID by its URL.

        Args:
            url (str): The URL of the file.

        Returns:
            int: The file ID associated with the URL.
        """
        return self.data_registry_contract.functions.getFileIdByUrl(url).call()

    def get_file(self, file_id: int):
        """Get the file information according to the file id."""
        return self.data_registry_contract.functions.getFile(file_id).call()

    def get_file_permission(self, file_id: int, account: str):
        """Get the encryption key for the account."""
        return self.data_registry_contract.functions.getFilePermission(
            file_id, account
        ).call()

    def get_file_proof(self, file_id: int, index: int):
        """Get the file proof."""
        return self.data_registry_contract.functions.getFileProof(file_id, index).call()

    def get_files_count(self):
        """Get the file total count."""
        return self.data_registry_contract.functions.filesCount().call()

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

    def node_fee(self) -> int:
        return self.verified_computing_contract.functions.nodeFee().call()

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
        message_hash = Web3.keccak(data.abi_encode())
        eth_message = encode_defunct(primitive=message_hash)
        signature = self.w3.eth.account.sign_message(
            eth_message, self.wallet.key
        ).signature.hex()

        proof = {
            "signature": Web3.to_bytes(hexstr=HexBytes(signature).hex()),
            "data": {
                "id": data.id,
                "score": data.score,
                "fileUrl": data.file_url,
                "proofUrl": data.proof_url,
            },
        }

        return self.send_transaction(
            self.data_registry_contract.functions.addProof(file_id, proof)
        )

    def get_job(self, job_id: int):
        return self.verified_computing_contract.functions.getJob(job_id).call()

    def file_job_ids(self, file_id: int):
        return self.verified_computing_contract.functions.fileJobIds(file_id).call()

    def jobs_count(self) -> int:
        return self.verified_computing_contract.functions.jobsCount().call()

    def node_list_at(self, index: int):
        return self.verified_computing_contract.functions.nodeListAt(index).call()

    def active_node_list(self) -> List[str]:
        return self.verified_computing_contract.functions.activeNodeList().call()

    def active_node_list_at(self, index: int):
        return self.verified_computing_contract.functions.activeNodeListAt(index).call()

    def nodes_count(self) -> int:
        return self.verified_computing_contract.functions.nodesCount().call()

    def active_nodes_count(self) -> int:
        return self.verified_computing_contract.functions.activeNodesCount().call()

    def is_node(self, address: str) -> bool:
        return self.verified_computing_contract.functions.isNode(address).call()

    def submit_job(self, file_id: int, value: int):
        return self.send_transaction(
            self.verified_computing_contract.functions.submitJob(file_id), value=value
        )

    def claim(self):
        return self.send_transaction(self.verified_computing_contract.functions.claim())

    def request_reward(self, file_id: int, proof_index: int = 1):
        return self.send_transaction(
            self.data_registry_contract.functions.requestReward(file_id, proof_index)
        )

    def mint_dat(self, to: str, amount: int, token_uri: str, verified: bool):
        """Mint a new Data Anchor Token (DAT) with the specified parameters."""
        return self.send_transaction(
            self.data_anchoring_token_contract.functions.mint(
                to, amount, token_uri, verified
            )
        )

    def get_dat_balance(self, account: str, id: int):
        """Returns the balance of a specific Data Anchor Token (DAT) for a given account and token ID."""
        return self.data_anchoring_token_contract.functions.balanceOf(
            account, id
        ).call()

    def data_uri(self, token_id: int):
        """Returns the Uri for a specific Data Anchor Token (DAT) by its token ID."""
        return self.data_anchoring_token_contract.functions.uri(token_id).call()

    def get_user(self, user: str):
        """Get the user for the settlement."""
        return self.settlement_contract.functions.getUser(user).call()

    def get_all_users(self):
        """Get all users registered for the settlement."""
        return self.settlement_contract.functions.getAllUsers().call()

    def add_user(self, amount: int):
        return self.send_transaction(
            self.settlement_contract.functions.addUser(), value=amount
        )

    def delete_user(self):
        return self.send_transaction(
            self.settlement_contract.functions.deleteUser(),
        )

    def deposit(self, amount: int):
        return self.send_transaction(
            self.settlement_contract.functions.deposit(), value=amount
        )

    def withdraw(self, amount: int):
        return self.send_transaction(
            self.settlement_contract.functions.withdraw(amount)
        )

    def deposit_query(self, node: str, amount: int):
        return self.send_transaction(
            self.settlement_contract.functions.depositQuery(node, amount)
        )

    def deposit_inference(self, node: str, amount: int):
        return self.send_transaction(
            self.settlement_contract.functions.depositInference(node, amount)
        )

    def deposit_training(self, node: str, amount: int):
        return self.send_transaction(
            self.settlement_contract.functions.depositTraining(node, amount)
        )

    def retrieve_query(self, nodes: List[str]):
        return self.send_transaction(
            self.settlement_contract.functions.retrieveQuery(nodes)
        )

    def retrieve_inference(self, nodes: List[str]):
        return self.send_transaction(
            self.settlement_contract.functions.retrieveInference(nodes)
        )

    def retrieve_training(self, nodes: List[str]):
        return self.send_transaction(
            self.settlement_contract.functions.retrieveTraining(nodes)
        )

    def add_query_node(self, address: str, url: str, public_key: str):
        return self.send_transaction(
            self.query_contract.functions.addNode(address, url, public_key)
        )

    def remove_query_node(self, address: str):
        return self.send_transaction(self.query_contract.functions.removeNode(address))

    def get_query_node(self, address: str):
        return self.query_contract.functions.getNode(address).call()

    def query_node_list(
        self,
    ):
        return self.query_contract.functions.NodeList().call()

    def get_query_account(self, user: str, node: str):
        return self.query_contract.functions.getAccount(user, node).call()

    def query_settlement_fees(
        self,
        data: SettlementData,
    ):
        message_hash = Web3.keccak(data.abi_encode())
        eth_message = encode_defunct(primitive=message_hash)
        signature = self.w3.eth.account.sign_message(
            eth_message, self.wallet.key
        ).signature.hex()

        first_param = Web3.to_bytes(hexstr=signature)
        second_param = (
            data.id,
            data.user,
            data.cost,
            data.nonce,
            Web3.to_bytes(hexstr=data.user_signature),
        )
        contract_params = (first_param, second_param)
        return self.send_transaction(
            self.query_contract.functions.settlementFees(contract_params)
        )

    def add_inference_node(self, address: str, url: str, public_key: str):
        return self.send_transaction(
            self.inference_contract.functions.addNode(address, url, public_key)
        )

    def remove_inference_node(self, address: str):
        return self.send_transaction(
            self.inference_contract.functions.removeNode(address)
        )

    def get_inference_node(self, address: str):
        return self.inference_contract.functions.getNode(address).call()

    def inference_node_list(
        self,
    ):
        return self.inference_contract.functions.NodeList().call()

    def get_inference_account(self, user: str, node: str):
        return self.inference_contract.functions.getAccount(user, node).call()

    def inference_settlement_fees(
        self,
        data: SettlementData,
    ):
        message_hash = Web3.keccak(data.abi_encode())
        eth_message = encode_defunct(primitive=message_hash)
        signature = self.w3.eth.account.sign_message(
            eth_message, self.wallet.key
        ).signature.hex()

        first_param = Web3.to_bytes(hexstr=signature)
        second_param = (
            data.id,
            data.user,
            data.cost,
            data.nonce,
            Web3.to_bytes(hexstr=data.user_signature),
        )
        contract_params = (first_param, second_param)

        return self.send_transaction(
            self.inference_contract.functions.settlementFees(contract_params)
        )

    def add_training_node(self, address: str, url: str, public_key: str):
        return self.send_transaction(
            self.training_contract.functions.addNode(address, url, public_key)
        )

    def remove_training_node(self, address: str):
        return self.send_transaction(
            self.training_contract.functions.removeNode(address)
        )

    def get_training_node(self, address: str):
        return self.training_contract.functions.getNode(address).call()

    def training_node_list(
        self,
    ):
        return self.training_contract.functions.NodeList().call()

    def get_training_account(self, user: str, node: str):
        return self.training_contract.functions.getAccount(user, node).call()

    def training_settlement_fees(self, data: SettlementData):
        message_hash = Web3.keccak(data.abi_encode())
        eth_message = encode_defunct(primitive=message_hash)
        signature = self.w3.eth.account.sign_message(
            eth_message, self.wallet.key
        ).signature.hex()

        first_param = Web3.to_bytes(hexstr=signature)
        second_param = (
            data.id,
            data.user,
            data.cost,
            data.nonce,
            Web3.to_bytes(hexstr=data.user_signature),
        )
        contract_params = (first_param, second_param)

        return self.send_transaction(
            self.training_contract.functions.settlementFees(contract_params)
        )

    def get_request_headers(
        self, node: str, file_id: int | None = None, nonce: int | None = None
    ) -> Dict[str, str]:
        """Get the billing-related headers for the request for the AI node"""

        def _secure_nonce():
            current_time = int(time.time() * 1000)
            random_part = random.randint(0, 99999)
            nonce = current_time * 100000 + random_part
            return nonce

        return (
            SettlementRequest(
                nonce=nonce or _secure_nonce(),
                user=self.wallet.address,
                node=node,
                file_id=file_id,
            )
            .generate_signature(self.wallet.key)
            .to_request_headers()
        )
