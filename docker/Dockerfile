FROM nvcr.io/nvidia/pytorch:20.11-py3

RUN apt-get update && apt-get upgrade -y && apt-get install -y vim && apt-get install -y git
RUN apt-get install -y g++ && apt-get install -y openjdk-11-jdk && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install matplotlib seaborn scikit-learn scipy pandas numpy jupyter

# konlpy
RUN pip install konlpy   
RUN bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

RUN apt-get install -y fonts-nanum
RUN rm -rf ~/.cache/matplotlib/*

WORKDIR /

RUN pip freeze > requirements.txt

ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME


