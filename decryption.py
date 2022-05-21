from cryptography.fernet import Fernet

key= ''
with open('SekretnyKlucz1.key', 'rb') as file:
    key= file.read()

encrypted_data= ''
with open('zaszyfrowane_dane.py', 'rb') as file:
    encrypted_data= file.read()

f= Fernet(key)

decrypted_data= f.decrypt(encrypted_data)

print("Zaszyfrowane dane: ", encrypted_data.decode())

print()

print("Odszyfrowane dane: ", decrypted_data.decode())
