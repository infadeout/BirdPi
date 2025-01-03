FROM --platform=linux/arm64/v8 python:3.8-slim-buster

WORKDIR /app

ARG MODEL_FILE=BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite
ARG LABELS_FILE=labels_lang.txt
ARG MODEL_DIR=/app/model
ARG DB_DIR=/app/database
ARG DEBIAN_FRONTEND=noninteractive

ENV MODEL_PATH=${MODEL_DIR}/${MODEL_FILE}
ENV LABELS_PATH=${MODEL_DIR}/${LABELS_FILE}
ENV DB_PATH=${DB_DIR}/birds.db
ENV PORT=5050
ENV SERVER=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libsndfile1 \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p ${MODEL_DIR} ${DB_DIR}

# Copy server code and initialization
COPY ./recording/server.py /app/
COPY ./recording/utils /app/utils
COPY ./model/${MODEL_FILE} ${MODEL_PATH}
COPY ./model/${LABELS_FILE} ${LABELS_PATH}
COPY ./database/init.sql /app/

# Install Python packages
RUN pip install --no-cache-dir \
    numpy \
    librosa \
    tflite-runtime \
    soundfile \
    tzlocal \
    apprise

# Initialize database
RUN sqlite3 ${DB_PATH} < /app/init.sql

EXPOSE ${PORT}

CMD ["python", "server.py"]