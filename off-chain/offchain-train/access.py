import base64
import random
import requests
import json
import time
from Crypto.Cipher import PKCS1_v1_5
from Crypto import Random
from Crypto.PublicKey import RSA


# ------------------------生成密钥对------------------------
def create_rsa_pair(is_save=False):
    '''
    创建rsa公钥私钥对
    :param is_save: default:False
    :return: public_key, private_key
    '''
    f = RSA.generate(2048)
    private_key = f.exportKey("PEM")  # 生成私钥
    public_key = f.publickey().exportKey()  # 生成公钥
    if is_save:
        with open("crypto_private_key.pem", "wb") as f:
            f.write(private_key)
        with open("crypto_public_key.pem", "wb") as f:
            f.write(public_key)
    return public_key, private_key


def read_public_key(file_path="crypto_public_key.pem") -> bytes:
    with open(file_path, "rb") as x:
        b = x.read()
        return b


def read_private_key(file_path="crypto_private_key.pem") -> bytes:
    with open(file_path, "rb") as x:
        b = x.read()
        return b


# ------------------------加密------------------------
def encryption(text: str, public_key: bytes):
    # 字符串指定编码（转为bytes）
    text = text.encode('utf-8')
    # 构建公钥对象
    cipher_public = PKCS1_v1_5.new(RSA.importKey(public_key))
    # 加密（bytes）
    text_encrypted = cipher_public.encrypt(text)
    # base64编码，并转为字符串
    text_encrypted_base64 = base64.b64encode(text_encrypted).decode()
    return text_encrypted_base64


# ------------------------解密------------------------
def decryption(text_encrypted_base64: str, private_key: bytes):
    # 字符串指定编码（转为bytes）
    text_encrypted_base64 = text_encrypted_base64.encode('utf-8')
    # base64解码
    text_encrypted = base64.b64decode(text_encrypted_base64)
    # 构建私钥对象
    cipher_private = PKCS1_v1_5.new(RSA.importKey(private_key))
    # 解密（bytes）
    text_decrypted = cipher_private.decrypt(text_encrypted, Random.new().read)
    # 解码为字符串
    text_decrypted = text_decrypted.decode()
    return text_decrypted

def upload_to_chain(client_id, request_model_id, requester_public_key):
    x = random.randint(1, 999999)
    encrypt_pubk_x = encryption(x, requester_public_key)
    data = {
        "requester": client_id,
        "model_id": request_model_id,
        "random_number": encrypt_pubk_x,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    }

    headers = {'content-type': 'application/json',
               'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}


    org_server = "http://114.212.82.242:8080/"
    session = requests.Session()
    upload_access_request = "upload_to_chain"
    response = session.post(org_server + upload_access_request, data=json.dumps(data), headers=headers).content
    print(response)

    encrypt_prik_x = encryption(x, private_key)
    data = {
        "requester": client_id,
        "model_id": request_model_id,
        "random_number": encrypt_prik_x,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    }

    access_request2 = "model_request"
    response = session.post(org_server + access_request2, data=json.dumps(data), headers=headers).content
    print(response)

if __name__ == '__main__':
    # 生成密钥对
    # create_rsa_pair(is_save=True)
    # public_key = read_public_key()
    # private_key = read_private_key()
    public_key, private_key = create_rsa_pair(is_save=False)

    # 加密
    text = '123456'
    text_encrypted_base64 = encryption(text, public_key)
    print('密文：', text_encrypted_base64)

    # 解密
    text_decrypted = decryption(text_encrypted_base64, private_key)
    print('明文：', text_decrypted)


