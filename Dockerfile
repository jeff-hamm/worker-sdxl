#FROM runpod/base:0.4.2-cuda11.8.0
#FROM runpod_api_base:0.4.2-cuda11.8.0
ARG BASE_LABEL=0.4.2
ARG MODELS_LABEL=loaded_models-latest
ARG NO_MODELS
FROM jumpmyman/sdxl_gen:${MODELS_LABEL} as loaded_models

FROM jumpmyman/sdxl_gen:${BASE_LABEL}
ARG HF_HOME="/runpod-volume/.cache/huggingface" 
ARG HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
ENV HF_HOME=${HF_HOME}
ENV HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}
ENV HF_HUB_ENABLE_HF_TRANSFER=0
COPY --from=loaded_models /huggingface/hub/. ${HUGGINGFACE_HUB_CACHE}
COPY src/config.py /
COPY src/ModelHandler.py /
ARG MODEL_NAME="canny_img2img"
ENV MODEL_NAME=${MODEL_NAME}
COPY requirements.txt /
RUN python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    ${NO_MODELS} || python3.11 /ModelHandler.py --model_name="${MODEL_NAME}" && \
    ls $HUGGINGFACE_HUB_CACHE
ADD src .
ARG SRC_SERVER=https://raw.githubusercontent.com/jeff-hamm/worker-sdxl/main
ENV SRC_SERVER=${SRC_SERVER}
CMD curl -o /rp_handler.py $SRC_SERVER/src/rp_handler.py && \
    curl -o /rp_shemas.py $SRC_SERVER/src/rp_shemas.py && \
    curl -o /config.py $SRC_SERVER/src/config.py && \
    curl -o /ModelHandler.py $SRC_SERVER/src/ModelHandler.py && \
    python3.11 -u /rp_handler.py --model_name="${MODEL_NAME}"
