import base64
import json
import hashlib


def file_to_base64(file_path):
    with open(file_path, "rb") as file:
        # Read the file content
        file_content = file.read()
        # Encode the file content to Base64
        base64_content = base64.b64encode(file_content)
        # Decode the bytes to a UTF-8 string
        base64_string = base64_content.decode("utf-8")
        return base64_string


def json_2_sha256_key(json_data):
    json_string = json.dumps(json_data)
    sha256_key = hashlib.sha256(json_string.encode()).hexdigest()
    return sha256_key
