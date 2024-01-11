sudo docker run -it --gpus all \
    -v ~/Sam/Projects/dreamify/models:/app/models \
    -v ~/Sam/Projects/dreamify/people-pics:/app/people-pics \
    -v ~/Sam/Projects/dreamify/img2img-pics:/app/img2img-pics \
    -v ~/Sam/Projects/dreamify/finetuner.sh:/app/finetuner.sh \
    samuelhathcock/dreamify:1.56 bash