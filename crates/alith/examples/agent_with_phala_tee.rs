//! For the deplopment environment, set the environment variable `DSTACK_SIMULATOR_ENDPOINT` with the
//! simulator: https://github.com/Leechael/tappd-simulator/releases
//!
//! In production environments, mount the socket file in your docker container:
//! ```yaml
//! volumes:
//!   - /var/run/tappd.sock:/var/run/tappd.sock
//! ```
use alith::tee::phala::DstackClient;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = DstackClient::default();
    // Derive a key from a key path
    // Returns a key and a certificate chain
    println!(
        "Derive key: {:?}",
        client.derive_key(Some("test"), None, None).await?
    );
    // Get a TDX quote
    println!(
        "Generate report: {:?}",
        client.tdx_quote("test", Default::default()).await?
    );
    Ok(())
}
