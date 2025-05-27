import gnupg


def encrypt(data: bytes, password: str) -> bytes:
    gpg = gnupg.GPG()
    encrypted_data = gpg.encrypt(
        data, "", passphrase=password, symmetric=True, armor=False
    )
    return encrypted_data.data


def decrypt(data: bytes, password: str) -> bytes:
    gpg = gnupg.GPG()
    decrypted_data = gpg.decrypt(data, passphrase=password)
    return decrypted_data.data
