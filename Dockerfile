# 
FROM python:3.9

# 
WORKDIR /code

# 
# COPY ./requirements_docker.txt /code/requirements.txt
COPY ./install_docker.sh /code/install_docker.sh

# 
RUN ./install_docker.sh

# 
COPY ./ /code/app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]