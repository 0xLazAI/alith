import rsa

(pub_key, priv_key) = rsa.newkeys(3072)
pub_key_pem = pub_key.save_pkcs1().decode("utf-8")
priv_key_pem = priv_key.save_pkcs1().decode("utf-8")
print("Public Key (PEM):")
print(pub_key_pem)
print("\nPrivate Key (PEM):")
print(priv_key_pem)
