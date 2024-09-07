ARG BASE_VERSION=0.6.1
ARG SDXL=true
ARG CONTROLNET=true
ARG REVISION=true
ARG REFINER=true
FROM runpod/base:${BASE_VERSION}-cuda11.8.0 as runpod_base
#ENV HF_HUB_ENABLE_HF_TRANSFER=0
# Install Python dependencies (Worker Template)
RUN python3.11 -m pip install --upgrade pip
COPY base/requirements.txt /requirements.txt
COPY base/whl/. /whl/
RUN python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm -rf /whl && \
    rm -rf /requirements.txt

FROM runpod_base as 

FROM runpod_base as sdxl_base
COPY base/huggingface/. /root/.cache/huggingface/hub/.
COPY src/ModelHandler.py /ModelHandler.py
RUN echo "mv /root/.cache/huggingface/hub $HF_HOME" && \
    mkdir -p $HF_HOME && \
    mv /root/.cache/huggingface/hub $HF_HOME && \
    ls $HF_HOME && \
     python3.11 /ModelHandler.py && \
    rm -rf /ModelHandler.py
