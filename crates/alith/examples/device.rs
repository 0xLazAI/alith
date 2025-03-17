use alith::devices::DeviceConfig;

fn main() -> Result<(), anyhow::Error> {
    let mut device = DeviceConfig::default();
    device.initialize()?;
    println!("{}", device);
    Ok(())
}
