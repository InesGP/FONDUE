FROM python:3.9.19-bullseye

COPY . /FONDUE
WORKDIR /FONDUE
RUN pip install -r requirements.txt -U && pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

ENTRYPOINT [ "/bin/bash" ]