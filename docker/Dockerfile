FROM nvcr.io/nvidia/pytorch:20.11-py3

RUN apt-get update
RUN apt-get update && apt-get upgrade -y && apt-get install -y vim && apt-get install -y git
RUN apt-get install -y g++ && apt-get install -y curl

# java
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get update
RUN apt-get install -y openjdk-11-jdk

# konlpy
RUN pip install konlpy   
RUN bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

# requirements
RUN pip install --upgrade pip
RUN pip install matplotlib seaborn scikit-learn scipy pandas numpy jupyter gluonnlp==0.10.0 wandb==0.12.18 transformers==4.18.0 einops
RUN pip install git+https://git@github.com/SKTBrain/KoBERT.git@master

RUN apt-get install -y fonts-nanum
RUN rm -rf ~/.cache/matplotlib/*

# remove workspace
RUN rm -r /workspace/*


# download word-embeddings
WORKDIR /
RUN wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FeGIbSz2E1A63JZP_XIxnGaSRt7AhXFf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FeGIbSz2E1A63JZP_XIxnGaSRt7AhXFf" -O word-embeddings.zip && rm -rf ~/cookies.txt
RUN unzip word-embeddings.zip
RUN rm word-embeddings.zip


# download part1,2 checkpoints
RUN mkdir /saved_model \
    && mkdir /saved_model/Part1 \
    && mkdir /saved_model/Part2 \
    && wget https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part1/HAND.zip \
    && wget https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part1/FNDNet.zip \
    && wget https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part1/BERT.zip \
    && unzip HAND.zip -d /saved_model/Part1 \
    && unzip FNDNet.zip -d /saved_model/Part1 \
    && unzip BERT.zip -d /saved_model/Part1 \
    && rm *.zip \
    && wget https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part2/BERT.zip \
    && wget https://github.com/TooTouch/Fake-News-Detection-Dataset/releases/download/part2/KoBERTSeg.zip \
    && unzip BERT.zip -d /saved_model/Part2 \
    && unzip KoBERTSeg.zip -d /saved_model/Part2 \
    && rm *.zip

# clone fake news github
WORKDIR /workspace
RUN git clone https://github.com/TooTouch/Fake-News-Detection-Dataset.git