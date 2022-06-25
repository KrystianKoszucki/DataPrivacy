from cryptography.fernet import Fernet

key= ''
with open('SecretKey.key', 'rb') as file:
    key= file.read()

data= ''
with open('data_privacy.py', 'rb') as file:
    data= file.read()

f= Fernet(key)

encrypted_data= f.encrypt(data)

with open('encyprted_data.py', 'wb') as file:
    file.write(encrypted_data)
