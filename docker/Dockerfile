FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt update -y && apt install -y $(cat requirements_os.txt|grep -v '#')

RUN pip install -r ../requirements.txt
