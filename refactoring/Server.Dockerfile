# Build arguments and environment variables
ARG MODEL_FILE=BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite
ARG MODEL_DIR=/app/model
ARG DB_DIR=/app/database

# Set environment variables using ARGs
ENV MODEL_PATH=${MODEL_DIR}/${MODEL_FILE}
ENV LABELS_PATH=${MODEL_DIR}/labels.txt
ENV DB_PATH=${DB_DIR}/birds.db
ENV PORT=5050
ENV SERVER=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create directories using ARGs
RUN mkdir -p ${MODEL_DIR} ${DB_DIR}

# Copy server code and model using ARGs
COPY ./recording/server.py /app/
COPY ./model/${MODEL_FILE} ${MODEL_PATH}
COPY ./model/labels.txt ${MODEL_DIR}/

# Install Python packages
RUN pip install numpy librosa tflite-runtime soundfile

# Expose port using existing ENV
EXPOSE ${PORT}

CMD ["python", "server.py"]