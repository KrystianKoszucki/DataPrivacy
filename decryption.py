from cryptography.fernet import Fernet

key= ''
with open('SecretKey.key', 'rb') as file:
    key= file.read()

encrypted_data= ''
with open('encrypted_data.py', 'rb') as file:
    encrypted_data= file.read()

f= Fernet(key)

decrypted_data= f.decrypt(encrypted_data)

print("Encrypted data: ", encrypted_data.decode())

print()

print("Decrypted data: ", decrypted_data.decode())
