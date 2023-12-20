#!/bin/bash

vitual_env_path=".venv"
required_version="3.10.12"
current_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')

# Colors
BBlack='\033[1;30m'       # Black
BRed='\033[1;31m'         # Red
BGreen='\033[1;32m'       # Green
BYellow='\033[1;33m'      # Yellow
BBlue='\033[1;34m'        # Blue
BPurple='\033[1;35m'      # Purple
BCyan='\033[1;36m'        # Cyan
BWhite='\033[1;37m'       # White
Color_Off='\033[0m'       # Text Reset

if [[ $(printf "$required_version\n$current_version" | sort -V | head -n 1) == $required_version ]]; then
    printf "${BGreen}Creating the virtual environment...${Color_Off}\n"
    python -m venv $vitual_env_path

    printf "${BGreen}Deactivating the current environment...${Color_Off}\n"
    deactivate  # Deactivate the current virtual environment

    printf "${BGreen}Activating the new environment...${Color_Off}\n"
    source $vitual_env_path/bin/activate

    printf "${BGreen}Upgrading pip...${Color_Off}\n"
    pip install --upgrade pip

    printf "${BGreen}Installing the Python packages${Color_Off}\n"
    pip install pycaret -q
    pip install pydantic==1.10 fastapi python-multipart matplotlib fastapi scipy streamlit uvicorn -q

    printf "${BPurple}Execute this command:${Color_Off}\n source $vitual_env_path/bin/activate\n\n"
else
    echo "Python version is $current_version, which is equal to or higher than $required_version"
    echo "Please install the python $required_version and execute this script again."
fi
