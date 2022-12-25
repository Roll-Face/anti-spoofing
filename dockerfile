FROM python:3.9
USER root

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install ffmpeg libavcodec-extra -y && \
    apt-get update
RUN python3 -m pip install --upgrade pip

RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install pandas
RUN pip3 install torch torchvision torchaudio

WORKDIR /zalo_challenge
COPY . zalo_challenge

CMD /bin/bash