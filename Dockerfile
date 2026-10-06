FROM gooseai/torch-base:6cfdc11
RUN apt-get install -y cuda-nvcc-11-3 cuda-nvml-dev-11-3 libcurand-dev-11-3 \
                       libcublas-dev-11-3 libcusparse-dev-11-3 \
                       libcusolver-dev-11-3 cuda-nvprof-11-3 \
                       ninja-build && \
    apt-get clean
RUN mkdir /app
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
COPY accelerate_config.yaml .
COPY finetuner.py .
COPY gen_imgs.py .
COPY mass_generate.py .
COPY mass_generate.sh .
COPY embeddings embeddings
COPY img2img-pics img2img-pics
# COPY class-images class-images
COPY templates.json .

RUN pip3 install --no-cache-dir -r requirements.txt

# RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
# RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/include
# RUN export PATH="/usr/local/cuda-11.8/bin:$PATH"
# RUN pip install ninja
# RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

RUN mkdir -p /root/.deepface/weights
RUN wget https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
RUN mv vgg_face_weights.h5 /root/.deepface/weights/

# # Start with an OS -> Install run environment -> copy app files -> run app
# FROM gooseai/torch-base:6cfdc11
# # Update CUDA to version 11.8 (please verify the package names)
# # RUN apt-get install -y cuda-nvcc-11-8 cuda-nvml-dev-11-8 libcurand-dev-11-8 \
# #                        libcublas-dev-11-8 libcusparse-dev-11-8 \
# #                        libcusolver-dev-11-8 cuda-nvprof-11-8 \
# #                        ninja-build && \
# #     apt-get clean 
    
# RUN apt-get update && \
# 	    apt-get install -y cmake && \
# 	    rm -rf /var/lib/apt/lists/*
# RUN mkdir /app
# WORKDIR /app
# RUN mkdir results
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# COPY environment.yml .
# COPY accelerate_config.yaml .
# COPY finetuner.py .
# COPY finetuner.sh .
# COPY gen_imgs.py .
# COPY mass_generate.py .
# COPY mass_generate.sh .
# COPY embeddings embeddings
# COPY img2img-pics img2img-pics
# COPY templates.json .

# # Replace pip installation with Conda and the environment file
# RUN apt-get install -y cuda
# RUN conda env create -f environment.yml

# # Add source activate myenv to .bashrc so the environment is activated every time a new shell is opened
# RUN echo "source activate myenv" >> ~/.bashrc

# RUN mkdir -p /root/.deepface/weights
# RUN wget https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
# RUN mv vgg_face_weights.h5 /root/.deepface/weights/

# # Set LD_LIBRARY_PATH for CUDA
# ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}



