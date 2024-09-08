ARG API_BASE_IMAGE=runpod/ai-api-sdxl
ARG API_BASE_VERSION=1.2.1
ARG BASE_VERSION=0.4.2

FROM runpod/base:${BASE_VERSION}-cuda11.8.0 AS runpod_base
#FROM ${BASE_IMAGE}:${BASE_VERSION} as runpod_base
ENV HF_HUB_ENABLE_HF_TRANSFER=0
# Install Python dependencies (Worker Template)
RUN python3.11 -m pip install --upgrade pip
COPY base/requirements.txt /requirements.txt
COPY base/whl/. /whl/
RUN python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm -rf /whl && \
    rm -rf /requirements.txt
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ARG HF_HOME="/runpod-volume/.cache/huggingface" 
ENV HF_HOME="${HF_HOME}/"
ARG HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
ENV HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}/"
ENV PYTHON_VERSION=3.11
RUN mkdir -p $HUGGINGFACE_HUB_CACHE

# FROM runpod_base as sdxl_base
# ARG HUGGINGFACE_HUB_CACHE
# COPY --from=sdxl_loader_final ${HUGGINGFACE_HUB_CACHE} ${HUGGINGFACE_HUB_CACHE}
# RUN python3.11 /ModelHandler.py && \
#     rm -rf /ModelHandler.py
