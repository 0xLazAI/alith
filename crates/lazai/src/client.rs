use alloy::{
    contract::{CallBuilder, CallDecoder},
    network::{Ethereum, EthereumWallet, TransactionBuilder, TransactionBuilderError},
    primitives::{Address, U256, keccak256},
    providers::Provider,
    sol_types::SolValue,
};
use std::ops::Deref;
use thiserror::Error;

use crate::{
    ChainConfig, ChainError, ChainManager, Proof, ProofData, Wallet, WalletError,
    chain::AlloyProvider,
    contracts::{
        ContractConfig, File, IDataRegistry::IDataRegistryInstance,
        IVerifiedComputing::IVerifiedComputingInstance, NodeInfo, Permission,
    },
};

#[derive(Debug, Clone)]
pub struct Client {
    manager: ChainManager,
    pub config: ContractConfig,
}

impl Client {
    /// New the LazAI client with the chain config and wallet.
    pub fn new(
        wallet: Wallet,
        chain_config: ChainConfig,
        contract_config: ContractConfig,
    ) -> Result<Self, ClientError> {
        let manager = ChainManager::new(chain_config, wallet)?;
        Ok(Self {
            manager,
            config: contract_config,
        })
    }

    /// New the default LazAI client.
    pub fn new_default() -> Result<Self, ClientError> {
        Ok(Self {
            manager: ChainManager::new_default()?,
            config: ContractConfig::default(),
        })
    }

    /// Add privacy data url and encrypted key base64 format into the data registry on LazAI.
    pub async fn add_file(&self, url: impl AsRef<str>) -> Result<U256, ClientError> {
        let contract = self.data_registry_contract();
        self.send_transaction(
            contract.addFile(url.as_ref().to_string()),
            self.config.data_registry_address,
            None,
        )
        .await?;

        self.get_file_id_by_url(url).await
    }

    /// Add privacy data url into the data registry on LazAI.
    pub async fn add_file_with_permissions(
        &self,
        url: impl AsRef<str>,
        owner: Address,
        permissions: Vec<Permission>,
    ) -> Result<U256, ClientError> {
        let contract = self.data_registry_contract();
        self.send_transaction(
            contract.addFileWithPermissions(url.as_ref().to_string(), owner, permissions),
            self.config.data_registry_address,
            None,
        )
        .await?;

        self.get_file_id_by_url(url).await
    }

    /// Add the permission for the file.
    pub async fn add_permission_for_file(
        &self,
        file_id: U256,
        permission: Permission,
    ) -> Result<(), ClientError> {
        let contract = self.data_registry_contract();
        self.send_transaction(
            contract.addPermissionForFile(file_id, permission.account, permission.key),
            self.config.data_registry_address,
            None,
        )
        .await?;
        Ok(())
    }

    /// Get the file id by the url.
    pub async fn get_file_id_by_url(&self, url: impl AsRef<str>) -> Result<U256, ClientError> {
        let contract = self.data_registry_contract();
        let builder = self
            .call_builder(
                contract.getFileIdByUrl(url.as_ref().to_string()),
                self.config.data_registry_address,
                None,
            )
            .await?;
        let file_id = builder
            .call()
            .await
            .map_err(|err| ClientError::ContractCallError(err.to_string()))?;
        Ok(file_id)
    }

    /// Get the file information according to the file id on LazAI.
    pub async fn get_file(&self, file_id: U256) -> Result<File, ClientError> {
        let contract = self.data_registry_contract();
        let builder = self
            .call_builder(
                contract.getFile(file_id),
                self.config.data_registry_address,
                None,
            )
            .await?;
        let file = builder
            .call()
            .await
            .map_err(|err| ClientError::ContractCallError(err.to_string()))?;
        Ok(file)
    }

    /// Get the encryption key for the account on LazAI.
    pub async fn get_file_permission(
        &self,
        file_id: U256,
        account: Address,
    ) -> Result<String, ClientError> {
        let contract = self.data_registry_contract();
        let builder = self
            .call_builder(
                contract.getFilePermission(file_id, account),
                self.config.data_registry_address,
                None,
            )
            .await?;
        let key = builder
            .call()
            .await
            .map_err(|err| ClientError::ContractCallError(err.to_string()))?;
        Ok(key)
    }

    /// Get the file proof on LazAI.
    pub async fn get_file_proof(&self, file_id: U256, index: U256) -> Result<Proof, ClientError> {
        let contract = self.data_registry_contract();
        let builder = self
            .call_builder(
                contract.getFileProof(file_id, index),
                self.config.data_registry_address,
                None,
            )
            .await?;
        let proof = builder
            .call()
            .await
            .map_err(|err| ClientError::ContractCallError(err.to_string()))?;
        Ok(proof)
    }

    /// Get the file total count.
    pub async fn get_files_count(&self) -> Result<U256, ClientError> {
        let contract = self.data_registry_contract();
        let builder = self
            .call_builder(
                contract.getFilesCount(),
                self.config.data_registry_address,
                None,
            )
            .await?;
        let count = builder
            .call()
            .await
            .map_err(|err| ClientError::ContractCallError(err.to_string()))?;
        Ok(count)
    }

    pub async fn add_proof(&self, file_id: U256, data: ProofData) -> Result<(), ClientError> {
        let packed_data = keccak256(data.abi_encode());
        let signature = self.wallet.sign_message(packed_data.as_slice()).await?;
        let proof = Proof {
            signature: signature.as_bytes().to_vec().into(),
            data,
        };
        let contract = self.data_registry_contract();
        self.send_transaction(
            contract.addProof(file_id, proof),
            self.config.data_registry_address,
            None,
        )
        .await?;

        Ok(())
    }

    pub async fn add_node(
        &self,
        address: Address,
        url: impl AsRef<str>,
        public_key: impl AsRef<str>,
    ) -> Result<(), ClientError> {
        let contract = self.verified_computing_contract();
        self.send_transaction(
            contract.addNode(
                address,
                url.as_ref().to_string(),
                public_key.as_ref().to_string(),
            ),
            self.config.verified_computing_address,
            None,
        )
        .await?;

        Ok(())
    }

    pub async fn get_node(&self, address: Address) -> Result<Option<NodeInfo>, ClientError> {
        let contract = self.verified_computing_contract();
        let builder = self
            .call_builder(
                contract.getNode(address),
                self.config.verified_computing_address,
                None,
            )
            .await?;
        let info = builder
            .call()
            .await
            .map_err(|err| ClientError::ContractCallError(err.to_string()))?;
        if info.url.is_empty() {
            Ok(None)
        } else {
            Ok(Some(info))
        }
    }

    /// Get the node address list
    pub async fn node_list(&self) -> Result<Vec<Address>, ClientError> {
        let contract = self.verified_computing_contract();
        let builder = self
            .call_builder(
                contract.nodeList(),
                self.config.verified_computing_address,
                None,
            )
            .await?;
        let list = builder
            .call()
            .await
            .map_err(|err| ClientError::ContractCallError(err.to_string()))?;
        Ok(list)
    }

    /// Claim any rewards for node validators
    pub async fn claim(&self) -> Result<(), ClientError> {
        let contract = self.verified_computing_contract();
        self.send_transaction(
            contract.claim(),
            self.config.verified_computing_address,
            None,
        )
        .await?;

        Ok(())
    }

    #[inline]
    async fn call_builder<'a, D: CallDecoder>(
        &'a self,
        call_builder: CallBuilder<&'a &'a AlloyProvider, D>,
        to: Address,
        value: Option<U256>,
    ) -> Result<CallBuilder<&'a &'a AlloyProvider, D>, ClientError> {
        Ok(call_builder
            .from(self.wallet.address)
            .to(to)
            .chain_id(self.manager.config.chain_id)
            .value(value.unwrap_or_default())
            .nonce(self.get_nonce().await?))
    }

    async fn send_transaction<'a, D: CallDecoder>(
        &'a self,
        call_builder: CallBuilder<&'a &'a AlloyProvider, D>,
        to: Address,
        value: Option<U256>,
    ) -> Result<(), ClientError> {
        let builder = call_builder
            .from(self.wallet.address)
            .to(to)
            .chain_id(self.manager.config.chain_id)
            .value(value.unwrap_or_default())
            .nonce(self.get_nonce().await?);
        let tx = builder.into_transaction_request();
        let gas_limit = self.estimate_tx_gas(tx.clone()).await?;
        let gas_price = self.get_gas_price().await?;
        let priority_fee = self.get_max_priority_fee_per_gas().await?;
        let tx = tx
            .with_gas_limit(gas_limit)
            .with_max_fee_per_gas(gas_price)
            .with_max_priority_fee_per_gas(priority_fee);

        let wallet = EthereumWallet::new(self.wallet.signer.clone());
        let tx_envelope = tx
            .build(&wallet)
            .await
            .map_err(|err| ChainError::SigningError(err.to_string()))?;

        let _ = self
            .provider
            .send_tx_envelope(tx_envelope)
            .await
            .map_err(Into::<ChainError>::into)?;
        Ok(())
    }

    #[inline]
    fn data_registry_contract(&self) -> IDataRegistryInstance<&AlloyProvider> {
        IDataRegistryInstance::new(self.config.data_registry_address, &self.manager.provider)
    }

    #[inline]
    fn verified_computing_contract(&self) -> IVerifiedComputingInstance<&AlloyProvider> {
        IVerifiedComputingInstance::new(
            self.config.verified_computing_address,
            &self.manager.provider,
        )
    }
}

impl AsRef<ChainManager> for Client {
    fn as_ref(&self) -> &ChainManager {
        &self.manager
    }
}

impl AsMut<ChainManager> for Client {
    fn as_mut(&mut self) -> &mut ChainManager {
        &mut self.manager
    }
}

impl Deref for Client {
    type Target = ChainManager;

    fn deref(&self) -> &Self::Target {
        &self.manager
    }
}

#[derive(Error, Debug)]
pub enum ClientError {
    #[error("Chain error: {0}")]
    ChainError(#[from] ChainError),
    #[error("Tx build error: {0}")]
    TxBuildError(#[from] TransactionBuilderError<Ethereum>),
    #[error("Wallet error: {0}")]
    WalletError(#[from] WalletError),
    #[error("Contract call error: {0}")]
    ContractCallError(String),
}
