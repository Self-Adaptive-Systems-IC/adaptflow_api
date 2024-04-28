import uvicorn
from dotenv import dotenv_values
import os
config = dotenv_values(".env")
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")


create_directory_if_not_exists("./tmp")
create_directory_if_not_exists("./tmp/files")
create_directory_if_not_exists("./tmp/loaded_models")
host = str(config["IP"])
port = int(config["PORT"])
print(host, port)
uvicorn.run("src.app:app", host=host, port=port)


    

