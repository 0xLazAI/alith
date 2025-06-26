import rsa
import base64

(pub_key, priv_key) = rsa.newkeys(3072)
pub_key_pem = pub_key.save_pkcs1().decode("utf-8")
priv_key_pem = priv_key.save_pkcs1().decode("utf-8")
print("Public Key (PEM):")
print(pub_key_pem)
print("\nPrivate Key (PEM):")
print(priv_key_pem)
pub_key_base64 = base64.b64encode(pub_key_pem.encode("utf-8")).decode("utf-8")
priv_key_base64 = base64.b64encode(priv_key_pem.encode("utf-8")).decode("utf-8")
print("Public Key (PEM) base64:")
print(pub_key_base64)
print("\nPrivate Key (PEM) base64:")
print(priv_key_base64)
