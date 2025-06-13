import os

import rsa

from alith.data import decrypt, encrypt


def main():
    privacy_data = b"Hello, Privacy Data with PGP!"
    password = os.urandom(32).hex()
    (pub_key, priv_key) = rsa.newkeys(3072)
    encrypted_key = rsa.encrypt(password.encode(), pub_key)
    encrypted_data = encrypt(privacy_data, password)
    try:
        decrypted_password = rsa.decrypt(encrypted_key, priv_key).decode()
    except rsa.DecryptionError:
        raise ValueError("RSA Failed")
    assert decrypted_password == password
    decrypted_data = decrypt(encrypted_data, decrypted_password)
    assert decrypted_data == privacy_data
    print("Crypto test successfully!")


main()
