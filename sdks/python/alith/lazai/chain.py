import time
from os import getenv
from threading import Lock
from typing import Any, Tuple

from eth_account.signers.local import LocalAccount
from web3 import Web3
from web3.exceptions import ContractLogicError
from web3.types import HexBytes, TxReceipt

DEVNET_NETWORK = "LazAI Devnet"
TESTNET_NETWORK = "LazAI Testnet"
LOCAL_CHAIN_ENDPOINT = "http://localhost:8545"
TESTNET_ENDPOINT = "https://lazai-testnet.metisdevops.link"
TESTNET_CHAINID: int = 133718


class ChainConfig:
    def __init__(self, network: str, endpoint: str, chain_id: int):
        self.network = network
        self.chain_endpoint = endpoint
        self.chain_id = chain_id
        self.gas_multiplier = 1.5
        self.max_retries = 3

    @classmethod
    def from_network(cls, network: str) -> "ChainConfig":
        if network == DEVNET_NETWORK:
            return cls(network, LOCAL_CHAIN_ENDPOINT, TESTNET_CHAINID)
        elif network == TESTNET_NETWORK:
            return cls(network, TESTNET_ENDPOINT, TESTNET_CHAINID)
        else:
            raise ValueError(f"Unsupported network: {network}")

    @classmethod
    def local(cls) -> "ChainConfig":
        return cls(DEVNET_NETWORK, LOCAL_CHAIN_ENDPOINT, TESTNET_CHAINID)

    @classmethod
    def testnet(cls) -> "ChainConfig":
        return cls(TESTNET_NETWORK, TESTNET_ENDPOINT, TESTNET_CHAINID)


class ChainManager:
    def __init__(
        self,
        config: ChainConfig = ChainConfig.testnet(),
        private_key: str = getenv("PRIVATE_KEY", ""),
    ):
        self.config = config
        self.w3: Web3 = Web3(Web3.HTTPProvider(config.chain_endpoint))
        self.wallet: LocalAccount = self.w3.eth.account.from_key(private_key)
        self._nonce_lock = Lock()

    @classmethod
    def default(cls) -> "ChainManager":
        return cls(ChainConfig.from_network(DEVNET_NETWORK))

    def get_current_block(self) -> int:
        return self.w3.eth.block_number

    def get_balance(self, address=None) -> int:
        return self.w3.eth.get_balance(address or self.wallet.address)

    def get_nonce(self, address=None) -> int:
        return self.w3.eth.get_transaction_count(address or self.wallet.address)

    def get_gas_price(self) -> int:
        return self.w3.eth.gas_price

    def get_max_priority_fee_per_gas(self) -> int:
        return self.w3.eth.max_priority_fee

    def transfer(
        self, to: str, value: int, gas: int = 21000, gas_price: int = None
    ) -> str:
        if gas_price is None:
            gas_price = self.get_gas_price()
        tx = {
            "to": to,
            "nonce": self.get_nonce(),
            "chainId": self.config.chain_id,
            "value": value,
            "gas": gas,
            "maxFeePerGas": gas_price,
            "maxPriorityFeePerGas": self.get_max_priority_fee_per_gas(),
        }
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.wallet.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        return tx_hash.hex()

    def estimated_gas(self, to: str, value: int, data, gas_price=None) -> int:
        tx = {
            "chainId": self.config.chain_id,
            "from": self.wallet.address,
            "to": to,
            "value": value,
            "gasPrice": gas_price or self.get_gas_price(),
            "nonce": self.get_nonce(),
            "data": data,
        }
        return self.w3.eth.estimate_gas(tx) * 2

    def _clear_pending_transactions(self, max_wait_time: int = 180):
        try:
            pending_nonce = self.w3.eth.get_transaction_count(
                self.wallet.address, "pending"
            )
            confirmed_nonce = self.w3.eth.get_transaction_count(
                self.wallet.address, "latest"
            )
            eth_transfer_gas = 21000
            if pending_nonce > confirmed_nonce:
                initial_pending = pending_nonce - confirmed_nonce
                highest_nonce = pending_nonce - 1
                base_gas_price = self.w3.eth.gas_price
                gas_multiplier = 5
                for nonce in range(confirmed_nonce, pending_nonce):
                    for attempt in range(3):
                        replacement_tx = {
                            "from": self.wallet.address,
                            "to": self.wallet.address,
                            "value": 0,
                            "nonce": nonce,
                            "gas": eth_transfer_gas,
                            "gasPrice": int(
                                base_gas_price * (gas_multiplier + attempt * 2)
                            ),
                            "chainId": self.config.chain_id,
                        }

                        try:
                            signed_tx = self.w3.eth.account.sign_transaction(
                                replacement_tx, self.wallet.key
                            )
                            self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                            break
                        except Exception as e:
                            if (
                                "replacement transaction underpriced" in str(e)
                                and attempt < 2
                            ):
                                continue
                            else:
                                break

                pending_remaining = initial_pending
                start_time = time.time()
                while time.time() - start_time < max_wait_time:
                    current_nonce = self.w3.eth.get_transaction_count(
                        self.wallet.address, "latest"
                    )
                    pending_remaining = (
                        self.w3.eth.get_transaction_count(
                            self.wallet.address, "pending"
                        )
                        - current_nonce
                    )

                    if current_nonce > highest_nonce:
                        return

                    if pending_remaining == 0:
                        return

                    time.sleep(5)  # Check every 5 seconds

        except Exception:
            pass

    def send_transaction(
        self,
        function: Any,
        value: int = 0,
        max_retries: int = 3,
        base_gas_multiplier: float = 1.5,
        timeout: int = 30,
        max_pending_transactions: int = 10,
    ) -> Tuple[HexBytes, TxReceipt]:
        pending_count = self.w3.eth.get_transaction_count(
            self.wallet.address, "pending"
        ) - self.w3.eth.get_transaction_count(self.wallet.address, "latest")
        if max_pending_transactions > 0 and pending_count > max_pending_transactions:
            with self._nonce_lock:
                current_pending_count = self.w3.eth.get_transaction_count(
                    self.wallet.address, "pending"
                ) - self.w3.eth.get_transaction_count(self.wallet.address, "latest")
                if current_pending_count > max_pending_transactions:
                    self._clear_pending_transactions()

        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                with self._nonce_lock:
                    nonce = self.w3.eth.get_transaction_count(
                        self.wallet.address, "pending"
                    )
                    gas_estimate = function.estimate_gas(
                        {
                            "from": self.wallet.address,
                            "value": value,
                            "chainId": self.config.chain_id,
                            "nonce": nonce,
                        }
                    )
                    gas_limit = int(gas_estimate * 2)
                    base_fee = self.w3.eth.get_block("latest")["baseFeePerGas"]
                    gas_multiplier = base_gas_multiplier * (1.5**retry_count)
                    priority_fee = self.w3.eth.max_priority_fee
                    max_fee_per_gas = int(base_fee * gas_multiplier) + priority_fee
                    max_priority_fee_per_gas = priority_fee
                    tx = function.build_transaction(
                        {
                            "from": self.wallet.address,
                            "value": value,
                            "gas": gas_limit,
                            "maxFeePerGas": max_fee_per_gas,
                            "maxPriorityFeePerGas": max_priority_fee_per_gas,
                            "nonce": nonce,
                            "chainId": self.config.chain_id,
                            "type": 2,
                        }
                    )

                    signed_tx = self.w3.eth.account.sign_transaction(
                        tx, self.wallet.key
                    )
                    tx_hash = self.w3.eth.send_raw_transaction(
                        signed_tx.raw_transaction
                    )
                    tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    return tx_hash, tx_receipt

            except ContractLogicError as e:
                error_msg = f"Transaction revert: {str(e)}"
                raise Exception(error_msg)

            except Exception as e:
                # Handle other errors (network, timeout etc)
                last_error = e
                retry_count += 1

                if retry_count < max_retries:
                    wait_time = 2**retry_count
                    time.sleep(wait_time)
                else:
                    raise

        raise Exception(
            f"Failed to send transaction after {max_retries} attempts: {str(last_error)}"
        )
