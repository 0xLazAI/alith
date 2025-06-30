use crate::settlement::SettlementTuple;
use crate::{Client, ClientError};
use alloy::{
    hex::{self, FromHexError},
    primitives::{Address, U256, keccak256},
    signers::Signature,
    sol_types::SolType,
};
use std::{collections::HashMap, num::ParseIntError};

pub const USER_HEADER: &str = "X-LazAI-User";
pub const NONCE_HEADER: &str = "X-LazAI-Nonce";
pub const SIGNATURE_HEADER: &str = "X-LazAI-Signature";
pub const TOKEN_ID_HEADER: &str = "X-LazAI-Token-ID";
pub const FILE_ID_HEADER: &str = "X-LazAI-File-ID";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestType {
    Query,
    Inference,
    Training,
}

impl Client {
    /// Validate the request user and signature in the request headers
    pub async fn validate_request(
        &self,
        headers: HashMap<String, String>,
        request_type: RequestType,
    ) -> Result<(), ClientError> {
        let user = headers
            .get(USER_HEADER)
            .ok_or(ClientError::ValidationError(
                "Missing X-LazAI-User header".to_string(),
            ))?
            .parse()
            .map_err(|e: FromHexError| ClientError::ValidationError(e.to_string()))?;
        let nonce = headers
            .get(NONCE_HEADER)
            .ok_or(ClientError::ValidationError(
                "Missing X-LazAI-Nonce header".to_string(),
            ))?
            .parse()
            .map_err(|e: ParseIntError| ClientError::ValidationError(e.to_string()))?;
        let signature = headers
            .get(SIGNATURE_HEADER)
            .ok_or(ClientError::ValidationError(
                "Missing X-LazAI-Signature header".to_string(),
            ))?;
        self.validate_account_and_signature(user, nonce, signature, request_type)
            .await
    }

    /// Validate the request user and signature with the user address, nonce and signature.
    pub async fn validate_account_and_signature(
        &self,
        user: Address,
        nonce: u64,
        signature: &str,
        request_type: RequestType,
    ) -> Result<(), ClientError> {
        let node = self.wallet.address;
        let account = match request_type {
            RequestType::Query => self.get_query_account(user, node).await?,
            RequestType::Inference => self.get_inference_account(user, node).await?,
            RequestType::Training => self.get_training_account(user, node).await?,
        };

        if account.user != user {
            return Err(ClientError::ValidationError(format!(
                "Account {user} does not exist or is unauthorized"
            )));
        }
        if U256::from(nonce) <= account.nonce {
            return Err(ClientError::ValidationError(format!(
                "Invalid nonce: {nonce}. Must be greater than last nonce: {}",
                account.nonce
            )));
        }

        let recovered_address = self.recover_address(nonce, user, node, signature)?;
        if recovered_address != user {
            return Err(ClientError::ValidationError(
                "Signature verification failed: address mismatch".to_string(),
            ));
        }

        Ok(())
    }

    fn recover_address(
        &self,
        nonce: u64,
        user: Address,
        node: Address,
        signature: &str,
    ) -> Result<Address, ClientError> {
        let encoded_data = SettlementTuple::abi_encode(&(U256::from(nonce), user, node));
        let message_hash = keccak256(encoded_data);
        let signature_bytes = hex::decode(signature)
            .map_err(|e: FromHexError| ClientError::ValidationError(e.to_string()))?;
        let signature = Signature::try_from(signature_bytes.as_slice())
            .map_err(|e| ClientError::ValidationError(e.to_string()))?;
        signature
            .recover_address_from_msg(message_hash)
            .map_err(|e| ClientError::ValidationError(e.to_string()))
    }
}
