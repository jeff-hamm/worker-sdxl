ARG REVISION=true
#base or turbo
ARG BASE_MODEL=base_turbo
#true or false
ARG REFINER=true
# base, lora or false
ARG CANNY=base


ARG BASE_MODEL_NAME=model_loader${BASE_MODEL:+"_"$BASE_MODEL}
# ARG CANNY=${CANNY:-false}
# ARG REFINER=${REFINER:-false}


FROM jumpmyman/sdxl_gen:cached_models AS cached_files

FROM jumpmyman/sdxl_gen:base AS model_loader
ENV HF_HOME="/huggingface/"
ENV HUGGINGFACE_HUB_CACHE="/huggingface/hub/"
COPY src/config.py /config.py
COPY src/ModelHandler.py /ModelHandler.py
ARG FILE=models--madebyollin--sdxl-vae-fp16-fix
COPY --from=cached_files /huggingface/hub/${FILE}* /huggingface/hub/${FILE}
RUN python3.11 /ModelHandler.py --model_name "vae"

FROM model_loader AS model_loader_base
ARG FILE=models--stabilityai--stable-diffusion-xl-base-1.0
COPY --from=cached_files /huggingface/hub/${FILE}* /huggingface/hub/${FILE}
RUN python3.11 /ModelHandler.py --model_name "text2img"


FROM model_loader AS model_loader_turbo
ARG FILE=models--stabilityai--sdxl-turbo
COPY --from=cached_files /huggingface/hub/${FILE}* /huggingface/hub/${FILE}
#RUN python3.11 /ModelHandler.py --model_name "text2img_turbo"

FROM model_loader AS model_loader_base_turbo
COPY --from=model_loader_base /huggingface/hub/. /huggingface/hub/
COPY --from=model_loader_turbo /huggingface/hub/. /huggingface/hub/

FROM ${BASE_MODEL_NAME} AS model_loader_refiner_false
FROM ${BASE_MODEL_NAME} AS model_loader_refiner_true
ARG FILE=models--stabilityai--stable-diffusion-xl-refiner-1.0
COPY --from=cached_files /huggingface/hub/${FILE}* /huggingface/hub/${FILE}
RUN python3.11 /ModelHandler.py --model_name "refiner"

FROM model_loader_refiner_${REFINER} AS model_loader_canny_false

FROM model_loader_refiner_${REFINER} AS model_loader_canny_base
ARG FILE=models--diffusers--controlnet-canny-sdxl-1.0
COPY --from=cached_files /huggingface/hub/${FILE}* /huggingface/hub/${FILE}
RUN python3.11 /ModelHandler.py --model_name "canny"
FROM  model_loader_refiner_${REFINER} AS model_loader_canny_lora
#COPY --from=cached_files /huggingface/hub/*canny-sdxl-1.0 /huggingface/hub/
RUN python3.11 /ModelHandler.py --model_name "canny_lora"

FROM model_loader_canny_${CANNY} AS model_loader_final