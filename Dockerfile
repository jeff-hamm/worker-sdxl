#FROM runpod/base:0.4.2-cuda11.8.0
#FROM runpod_api_base:0.4.2-cuda11.8.0
FROM runpod_sdxl_api_base:0.4.2-cuda11.8.0

COPY src/ModelHandler.py /
RUN python3.11 /ModelHandler.py

ENV HF_HUB_ENABLE_HF_TRANSFER=0
COPY requirements.txt /
RUN python3.11 -m pip install --upgrade -r /requirements.txt
ADD src .
ARG MODEL_TYPE="canny"
ENV MODEL_TYPE=${MODEL_TYPE}
ARG INPUT_TYPE="controlnet"
ENV INPUT_TYPE=${INPUT_TYPE}
CMD python3.11 -u /rp_handler.py --model_type="${MODEL_TYPE}" --input_type="${INPUT_TYPE}"
