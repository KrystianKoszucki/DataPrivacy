from cryptography.fernet import Fernet
key= Fernet.generate_key()

with open ('SekretnyKlucz1.key', 'wb') as file:
    file.write(key)
