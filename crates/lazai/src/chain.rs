pub use crate::wallet::{LocalEthWallet, WalletError};
use alloy::{
    network::{EthereumWallet, TransactionBuilder},
    primitives::{Address, ChainId, TxKind, U256},
    providers::{
        Identity, Provider, ProviderBuilder, RootProvider,
        fillers::{BlobGasFiller, ChainIdFiller, FillProvider, GasFiller, JoinFill, NonceFiller},
    },
    rpc::types::{BlockId, TransactionInput, TransactionReceipt, TransactionRequest},
    transports::{RpcError, TransportErrorKind},
};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use thiserror::Error;

pub const DEVNET_NETWORK: &str = "LazAI Devnet";
pub const TESTNET_NETWORK: &str = "LazAI Testnet";
pub const LOCAL_CHAIN_ENDPOINT: &str = "http://localhost:8545";
pub const TESTNET_ENDPOINT: &str = "https://lazi-testnet.metisdevops.link";
pub const TESTNET_CHAINID: ChainId = 133718;

pub type Wallet = LocalEthWallet;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChainConfig {
    pub network: String,
    pub chain_endpoint: String,
    pub chain_id: ChainId,
    pub gas_multiplier: f64,
    pub max_retries: u32,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self::local()
    }
}

impl ChainConfig {
    pub fn testnet() -> Self {
        Self {
            network: TESTNET_NETWORK.to_string(),
            chain_endpoint: TESTNET_ENDPOINT.to_string(),
            chain_id: TESTNET_CHAINID,
            gas_multiplier: 1.5,
            max_retries: 3,
        }
    }

    pub fn local() -> Self {
        Self {
            network: DEVNET_NETWORK.to_string(),
            chain_endpoint: LOCAL_CHAIN_ENDPOINT.to_string(),
            chain_id: TESTNET_CHAINID,
            gas_multiplier: 1.5,
            max_retries: 1,
        }
    }
}

#[derive(Error, Debug)]
pub enum ChainError {
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Signing error: {0}")]
    SigningError(String),
    #[error("Transaction failed: {0}")]
    TransactionError(String),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("Contract error: {0}")]
    ContractError(String),
    #[error("Wallet error: {0}")]
    WalletError(#[from] WalletError),
    #[error("Url error: {0}")]
    UrlError(#[from] url::ParseError),
    #[error("Rpc error: {0}")]
    RpcError(#[from] RpcError<TransportErrorKind>),
}

pub type AlloyProvider = FillProvider<
    JoinFill<
        Identity,
        JoinFill<GasFiller, JoinFill<BlobGasFiller, JoinFill<NonceFiller, ChainIdFiller>>>,
    >,
    RootProvider,
>;

#[derive(Debug, Clone)]
pub struct ChainManager {
    pub config: ChainConfig,
    pub wallet: Wallet,
    pub provider: AlloyProvider,
    nonce_lock: Arc<Mutex<u32>>,
}

impl ChainManager {
    pub fn new(config: ChainConfig, wallet: Wallet) -> Result<Self, ChainError> {
        let url = Url::parse(&config.chain_endpoint)?;
        let provider = ProviderBuilder::new().connect_http(url);

        Ok(Self {
            config,
            provider,
            wallet,
            nonce_lock: Default::default(),
        })
    }

    #[inline]
    pub fn new_default() -> Result<Self, ChainError> {
        Self::new(ChainConfig::default(), Wallet::from_env()?)
    }

    #[inline]
    pub async fn get_current_block(&self) -> Result<u64, ChainError> {
        Ok(self.provider.get_block_number().await?)
    }

    #[inline]
    pub async fn get_balance(&self, address: Address) -> Result<U256, ChainError> {
        Ok(self.provider.get_balance(address).await?)
    }

    #[inline]
    pub async fn get_nonce(&self) -> Result<u64, ChainError> {
        Ok(self
            .provider
            .get_transaction_count(self.wallet.address)
            .await?)
    }

    #[inline]
    pub async fn get_gas_price(&self) -> Result<u128, ChainError> {
        Ok(self.provider.get_gas_price().await?)
    }

    #[inline]
    pub async fn get_max_priority_fee_per_gas(&self) -> Result<u128, ChainError> {
        Ok(self.provider.get_max_priority_fee_per_gas().await?)
    }

    #[inline]
    pub async fn get_balance_with_block(
        &self,
        address: Address,
        block: Option<BlockId>,
    ) -> Result<U256, ChainError> {
        Ok(if let Some(block) = block {
            self.provider.get_balance(address).block_id(block)
        } else {
            self.provider.get_balance(address)
        }
        .await?)
    }

    #[inline]
    pub async fn transfer(
        &self,
        to: Address,
        value: U256,
        gas_limit: u64,
        gas_price: Option<u128>,
    ) -> Result<TransactionReceipt, ChainError> {
        let nonce = self.get_nonce().await?;

        let gas_price = match gas_price {
            Some(price) => price,
            None => self.provider.get_gas_price().await?,
        };
        let priority_fee = self.provider.get_max_priority_fee_per_gas().await?;

        let tx = TransactionRequest::default()
            .with_to(to)
            .with_nonce(nonce)
            .with_chain_id(self.config.chain_id)
            .with_value(value)
            .with_gas_limit(gas_limit)
            .with_max_fee_per_gas(gas_price)
            .with_max_priority_fee_per_gas(priority_fee);

        let wallet = EthereumWallet::new(self.wallet.signer.clone());
        let tx_envelope = tx
            .build(&wallet)
            .await
            .map_err(|err| ChainError::SigningError(err.to_string()))?;

        self.provider
            .send_tx_envelope(tx_envelope)
            .await?
            .get_receipt()
            .await
            .map_err(|err| ChainError::TransactionError(err.to_string()))
    }

    #[inline]
    pub async fn estimate_gas(
        &self,
        to: Address,
        value: U256,
        data: Option<Vec<u8>>,
    ) -> Result<u64, ChainError> {
        Ok(self
            .provider
            .estimate_gas(TransactionRequest {
                chain_id: Some(self.config.chain_id),
                from: Some(self.wallet.address),
                to: Some(TxKind::Call(to)),
                value: Some(value),
                input: TransactionInput::maybe_input(data.map(|d| d.into())),
                ..Default::default()
            })
            .await?
            * 2)
    }

    #[inline]
    pub async fn estimate_tx_gas(&self, tx: TransactionRequest) -> Result<u64, ChainError> {
        Ok(self.provider.estimate_gas(tx).await? * 2)
    }

    pub async fn clear_pending_transactions(&self) -> Result<(), ChainError> {
        {
            let mut _lock = self.nonce_lock.lock().unwrap();
            *_lock += 1;
        }

        // Get all pending transactions for the account
        let address = self.wallet.address;
        let pending_nonce = self
            .provider
            .get_transaction_count(address)
            .pending()
            .await?;
        let confirmed_nonce = self
            .provider
            .get_transaction_count(address)
            .latest()
            .await?;

        if pending_nonce <= confirmed_nonce {
            return Ok(());
        }

        let base_gas_price = self.provider.get_gas_price().await?;
        let eth_transfer_gas = 21000;
        let gas_multiplier = 5;

        for nonce in confirmed_nonce..pending_nonce {
            for attempt in 0..3 {
                let tx = TransactionRequest::default()
                    .with_to(self.wallet.address)
                    .with_nonce(nonce)
                    .with_chain_id(self.config.chain_id)
                    .with_value(U256::ZERO)
                    .with_gas_limit(eth_transfer_gas)
                    .with_gas_price(base_gas_price * (gas_multiplier + attempt * 2));

                let wallet = EthereumWallet::new(self.wallet.signer.clone());
                let tx_envelope = tx
                    .build(&wallet)
                    .await
                    .map_err(|err| ChainError::SigningError(err.to_string()))?;

                let _ = self.provider.send_tx_envelope(tx_envelope).await?;
            }
        }
        Ok(())
    }
}
