FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV PYTHONIOENCODING=utf-8
ENV PYTHONHASHSEED=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        jq \
        wget \
        vim \
        curl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
RUN pip install --no-cache-dir \
        certifi==2021.10.8 \
        charset-normalizer==2.0.7 \
        click==8.0.3 \
        cycler==0.10.0 \
        datasets==2.7.1 \
        einops==0.3.2 \
        filelock==3.3.1 \
        huggingface-hub==0.11.1 \
        idna==3.3 \
        japanize-matplotlib==1.1.3 \
        joblib==1.1.0 \
        jsonlines==3.1.0 \
        kiwisolver==1.3.2 \
        logzero==1.7.0 \
        matplotlib==3.4.3 \
        nltk==3.6.5 \
        numpy==1.21.3 \
        packaging==21.0 \
        pandas==1.3.4 \
        Pillow==8.4.0 \
        pyparsing==3.0.1 \
        python-dateutil==2.8.2 \
        pytz==2021.3 \
        PyYAML==6.0 \
        regex==2021.10.23 \
        requests==2.26.0 \
        sacremoses==0.0.46 \
        scipy==1.7.1 \
        seaborn==0.11.2 \
        six==1.16.0 \
        tokenizers==0.10.3 \
        torch==1.10.0 \
        tqdm==4.62.3 \
        transformers==4.11.3 \
        typing-extensions==3.10.0.2 \
        urllib3==1.26.7

# Download transformers models in advance
RUN python -c "from huggingface_hub import HfApi, HfFolder, Repository"

WORKDIR /codes
#COPY knowledge_neurons/ /codes/knowledge_neurons/
