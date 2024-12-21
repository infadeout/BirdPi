FROM python:3.8-slim-buster

WORKDIR /app

ARG MODEL_FILE=BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite
ARG LABELS_FILE=labels_lang.txt
ARG MODEL_DIR=/app/model
ARG DEBIAN_FRONTEND=noninteractive

ENV MODEL_PATH=${MODEL_DIR}/${MODEL_FILE}
ENV LABELS_PATH=${MODEL_DIR}/${LABELS_FILE}
ENV PORT=5050
ENV SERVER=0.0.0.0

# Install system dependencies including HDF5
RUN apt-get update && apt-get install -y \
    python3-pip \
    libsndfile1 \
    pkg-config \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create model directory
RUN mkdir -p ${MODEL_DIR}

# Copy server code and model
COPY ./recording/server.py /app/
COPY ./model/${MODEL_FILE} ${MODEL_PATH}
COPY ./model/${LABELS_FILE} ${LABELS_PATH}

# Install Python packages in correct order
RUN pip install --no-cache-dir \
    numpy \
    h5py \
    && pip install --no-cache-dir \
    librosa \
    tflite-runtime \
    soundfile \
    tzlocal \
    tensorflow

EXPOSE ${PORT}

CMD ["python", "server.py"]