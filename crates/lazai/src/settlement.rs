use crate::ClientError;
use crate::request::{FILE_ID_HEADER, NONCE_HEADER, SIGNATURE_HEADER, USER_HEADER};
use alith_data::wallet::LocalEthWallet;
use alloy::{
    primitives::{Address, U256, keccak256},
    sol,
    sol_types::SolType,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type SettlementTuple = sol! { tuple(uint256, address, address) };

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SettlementSignature {
    pub user: Address,
    pub nonce: u64,
    /// Hex-format signature
    pub signature: String,
    pub file_id: Option<U256>,
}

impl SettlementSignature {
    pub fn to_request_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert(USER_HEADER.to_string(), self.user.to_string());
        headers.insert(NONCE_HEADER.to_string(), self.nonce.to_string());
        headers.insert(SIGNATURE_HEADER.to_string(), self.signature.clone());

        if let Some(file_id) = self.file_id {
            headers.insert(FILE_ID_HEADER.to_string(), file_id.to_string());
        }

        headers
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SettlementRequest {
    pub nonce: u64,
    pub user: Address,
    pub node: Address,
    pub file_id: Option<U256>,
}

impl SettlementRequest {
    #[inline]
    pub fn abi_encode(&self) -> Vec<u8> {
        SettlementTuple::abi_encode(&(U256::from(self.nonce), self.user, self.node))
    }

    pub async fn generate_signature(
        &self,
        wallet: &LocalEthWallet,
    ) -> Result<SettlementSignature, ClientError> {
        let encoded_data = self.abi_encode();
        let message_hash = keccak256(encoded_data);
        let signature = wallet
            .sign_message_hex(&message_hash.0)
            .await
            .map_err(|e| ClientError::SigningError(e.to_string()))?;

        Ok(SettlementSignature {
            user: self.user,
            nonce: self.nonce,
            signature: format!("0x{}", signature),
            file_id: self.file_id,
        })
    }
}
