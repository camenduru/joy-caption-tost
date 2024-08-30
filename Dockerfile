FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    transformers==4.44.2 bitsandbytes==0.43.3 sentencepiece==0.2.0 runpod && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/config.json -d /content/llama -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/generation_config.json -d /content/llama -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/llama/model-00001-of-00004.safetensors -d /content/llama -o model-00001-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/llama/model-00002-of-00004.safetensors -d /content/llama -o model-00002-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/llama/model-00003-of-00004.safetensors -d /content/llama -o model-00003-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/llama/model-00004-of-00004.safetensors -d /content/llama -o model-00004-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/model.safetensors.index.json -d /content/llama -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/special_tokens_map.json -d /content/llama -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/tokenizer.json -d /content/llama -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/tokenizer_config.json -d /content/llama -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/siglip/config.json -d /content/siglip -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/siglip/model.safetensors -d /content/siglip -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/siglip/preprocessor_config.json -d /content/siglip -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/siglip/special_tokens_map.json -d /content/siglip -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/siglip/spiece.model -d /content/siglip -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/siglip/tokenizer_config.json -d /content/siglip -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/adapter/config.yaml -d /content/adapter -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/adapter/image_adapter.pt -d /content/adapter -o tokenizer_config.json

COPY ./worker_runpod.py /content/worker_runpod.py
WORKDIR /content
CMD python worker_runpod.py