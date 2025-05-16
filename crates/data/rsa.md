# RSA Key Management Guide

## Key Pair Creation Process

1. Initiate the GPG Key Generation Wizard

```shell
gpg --full-generate-key
```

2. Configure Key Parameters

- Select RSA encryption and signing combination (option 1)
- Set key length: It is recommended to use 3072 bits (balancing security and performance)
- Complete user identity information configuration (name/email)
- Set a private key protection password (at least 12 characters, including numbers and special symbols)

3. Key Generation Process

The system will automatically collect environmental random numbers to complete the creation of the key pair. This process may require physical interaction (keyboard typing/mouse movement) to accelerate entropy collection.

## Key Backup and Export

### Private Key Backup

```shell
gpg --armor --export-secret-keys your-email@example.com > rsa-private-key.asc
```

### Public Key Extraction

Retrieve the Key Database (Parse the long-format output, fields 5-6 are the complete Key ID)

```shell
gpg --list-keys --with-colons --keyid-format LONG
```

Look for the most recently created entry. Example format:

```
pub:rsa4096:1234ABCD1234ABCD:20230701::SC::Your Name <your-email@example.com>
```

Precisely Export the Public Key. For example:

```bash
gpg --armor --export 1234ABCD1234ABCD > rsa-public-key.asc
```

## Key Format Conversion

Perform Base64 encoding conversion to enhance transmission compatibility:

```shell
openssl base64 -in rsa-public-key.asc -out rsa-public-key.asc.b64
openssl base64 -in rsa-private-key.asc -out rsa-private-key.asc.b64 
```

## Decrypting a file

```shell
gpg --output original_file.png --decrypt encrypted_file.png
```

## Key Import

Symmetric Key Integration

```shell
gpg --import decrypted_symmetric_key.asc
```

Verify Import Status

```shell
gpg --list-secret-keys --keyid-format=0xLONG
```

## Security Practice Recommendations

1. Regularly rotate keys (suggested cycle: 180 days)
2. Use Hardware Security Modules (HSM) or TEE to store and deal root keys
3. Implement key usage audit logs
4. Configure key auto-expiration policies (`gpg --quick-set-expire`)
