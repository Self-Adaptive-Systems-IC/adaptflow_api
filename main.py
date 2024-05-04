import uvicorn
from dotenv import dotenv_values,load_dotenv
import os
load_dotenv()
# print(os.environ.get("IP"))
IP = os.getenv('IP', '0.0.0.0')
PORT = os.getenv('PORT', '99')


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")


create_directory_if_not_exists("./tmp")
create_directory_if_not_exists("./tmp/files")
create_directory_if_not_exists("./tmp/loaded_models")

print(f"{str(IP)}:{int(PORT)}")
uvicorn.run("src.app:app", host=str(IP), port=int(PORT))


    

